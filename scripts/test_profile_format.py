"""
Black Sheep Ventures - OASIS Profile Format Validation
Verifies:
1. Twitter profile generation conforms to CSV schema.
2. Reddit profile generation conforms to detailed JSON schema.
"""

import os
import sys
import json
import csv
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile


def test_profile_formats():
    """Test and validate OASIS agent profile schemas."""
    print("=" * 60)
    print("Black Sheep Ventures - Profile Format Verification")
    print("=" * 60)
    
    test_profiles = [
        OasisAgentProfile(
            user_id=0,
            user_name="test_user_123",
            name="Test User",
            bio="A test user for validation",
            persona="Test User is an enthusiastic participant in social discussions.",
            karma=1500,
            friend_count=100,
            follower_count=200,
            statuses_count=500,
            age=25,
            gender="male",
            mbti="INTJ",
            country="United States",
            profession="Software Engineer",
            interested_topics=["Technology", "Artificial Intelligence"],
            source_entity_uuid="test-uuid-123",
            source_entity_type="Professional",
        ),
        OasisAgentProfile(
            user_id=1,
            user_name="org_official_456",
            name="Official Organization",
            bio="Official account for Enterprise Intelligence",
            persona="This is an official institutional account that communicates official positions.",
            karma=5000,
            friend_count=50,
            follower_count=10000,
            statuses_count=200,
            profession="Organization",
            interested_topics=["Public Policy", "Announcements"],
            source_entity_uuid="test-uuid-456",
            source_entity_type="Enterprise",
        ),
    ]
    
    generator = OasisProfileGenerator.__new__(OasisProfileGenerator)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        twitter_path = os.path.join(temp_dir, "twitter_profiles.csv")
        reddit_path = os.path.join(temp_dir, "reddit_profiles.json")
        
        # Test Twitter CSV Format
        print("\n1. Testing Twitter Profile (CSV format)")
        print("-" * 40)
        generator._save_twitter_csv(test_profiles, twitter_path)
        
        with open(twitter_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        print(f"   File: {twitter_path}")
        print(f"   Rows: {len(rows)}")
        print(f"   Headers: {list(rows[0].keys())}")
        print(f"\n   Sample Row (Row 1):")
        for key, value in rows[0].items():
            print(f"     {key}: {value}")
        
        required_twitter_fields = ['user_id', 'user_name', 'name', 'bio', 
                                   'friend_count', 'follower_count', 'statuses_count', 'created_at']
        missing = set(required_twitter_fields) - set(rows[0].keys())
        if missing:
            print(f"\n   [ERROR] Missing required fields: {missing}")
        else:
            print(f"\n   [PASS] All required Twitter fields are valid.")
        
        # Test Reddit JSON Format
        print("\n2. Testing Reddit Profile (Detailed JSON format)")
        print("-" * 40)
        generator._save_reddit_json(test_profiles, reddit_path)
        
        with open(reddit_path, 'r', encoding='utf-8') as f:
            reddit_data = json.load(f)
        
        print(f"   File: {reddit_path}")
        print(f"   Entries: {len(reddit_data)}")
        print(f"   Fields: {list(reddit_data[0].keys())}")
        print(f"\n   Sample Record (Record 1):")
        print(json.dumps(reddit_data[0], ensure_ascii=False, indent=4))
        
        required_reddit_fields = ['realname', 'username', 'bio', 'persona']
        optional_reddit_fields = ['age', 'gender', 'mbti', 'country', 'profession', 'interested_topics']
        
        missing = set(required_reddit_fields) - set(reddit_data[0].keys())
        if missing:
            print(f"\n   [ERROR] Missing required fields: {missing}")
        else:
            print(f"\n   [PASS] All required Reddit fields are valid.")
        
        present_optional = set(optional_reddit_fields) & set(reddit_data[0].keys())
        print(f"   [INFO] Optional fields present: {present_optional}")
    
    print("\n" + "=" * 60)
    print("Verification completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    test_profile_formats()
