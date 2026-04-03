"""
Generate synthetic test data for fake profile detection

This script expands the test.csv dataset by generating realistic fake and genuine profiles.
Maintains the same structure and statistical distributions as the original data.

Usage:
    python generate_test_data.py --count 500
    python generate_test_data.py --count 500 --fake-ratio 0.7
    
    # Preserve original 10 rows + add 490 more
    python generate_test_data.py --count 500 --preserve-original 

    python generate_test_data.py --count 400 --output data/test/my_test.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import random
from datetime import datetime

# Common username patterns
PREFIXES = ['user', 'the', 'real', 'official', 'i_am', 'its', 'just', 'call_me', 'hey', 'mr', 'miss', 'xXx']
SUFFIXES = ['_01', '_99', '_2024', '_pro', '_official', '_real', '_fake', '_bot', 'xXx', '123', '456', '777', '999']
COMMON_NAMES = [
    'john', 'sarah', 'mike', 'emma', 'david', 'lisa', 'alex', 'maria', 'james', 'anna',
    'chris', 'kate', 'ryan', 'jenny', 'kevin', 'amy', 'brian', 'rachel', 'tom', 'laura',
    'steve', 'nicole', 'mark', 'jessica', 'paul', 'michelle', 'daniel', 'amanda', 'jason', 'melissa',
    'robert', 'emily', 'william', 'ashley', 'joseph', 'brittany', 'michael', 'samantha', 'david', 'stephanie'
]
RANDOM_WORDS = [
    'cool', 'happy', 'fun', 'super', 'mega', 'ultra', 'best', 'top', 'great', 'awesome',
    'love', 'life', 'live', 'dream', 'star', 'sky', 'moon', 'sun', 'ocean', 'mountain',
    'tech', 'gamer', 'coder', 'artist', 'music', 'photo', 'style', 'fashion', 'food', 'travel',
    'fitness', 'yoga', 'dance', 'sing', 'write', 'read', 'learn', 'teach', 'build', 'create'
]

def generate_username():
    """Generate a realistic username"""
    patterns = [
        lambda: f"{random.choice(COMMON_NAMES)}{random.randint(100, 9999)}",
        lambda: f"{random.choice(RANDOM_WORDS)}_{random.choice(COMMON_NAMES)}",
        lambda: f"{random.choice(COMMON_NAMES)}_{random.choice(RANDOM_WORDS)}",
        lambda: f"{random.choice(PREFIXES)}{random.choice(COMMON_NAMES)}",
        lambda: f"{random.choice(COMMON_NAMES)}{random.choice(SUFFIXES)}",
        lambda: f"{random.choice(RANDOM_WORDS)}{random.randint(1, 999)}",
        lambda: ''.join(random.choices(COMMON_NAMES, k=2)) + str(random.randint(1, 99)),
        lambda: f"{''.join(random.sample('abcdefghijklmnopqrstuvwxyz', 8))}{random.randint(1,99)}",
    ]
    return random.choice(patterns)()

def generate_genuine_profile(profile_id):
    """Generate a genuine-looking profile (more realistic metrics)"""
    followers = np.random.lognormal(5, 2)  # More followers for genuine
    friends = np.random.lognormal(4, 1.5)
    
    followers = int(np.clip(followers, 50, 50000))
    friends = int(np.clip(friends, 30, 10000))
    
    statuses = int(np.random.lognormal(6, 2))
    statuses = int(np.clip(statuses, 100, 50000))
    
    favourites = int(np.random.lognormal(5, 2))
    favourites = int(np.clip(favourites, 50, 30000))
    
    listed = int(np.random.lognormal(1, 1.5))
    listed = int(np.clip(listed, 0, 500))
    
    description_len = int(np.random.normal(80, 40))
    description_len = int(np.clip(description_len, 20, 160))
    
    follower_friend_ratio = followers / friends if friends > 0 else 0
    engagement_score = followers + friends + favourites + statuses
    
    utc_offsets = [0, 3600, 7200, -18000, -21600, -25200, -28800, 10800, 19800, 28800, 32400, 36000]
    
    return {
        'id': profile_id,
        'screen_name': generate_username(),
        'description_len': description_len,
        'followers_count': followers,
        'utc_offset': random.choice(utc_offsets),
        'friends_count': friends,
        'follower_friend_ratio': round(follower_friend_ratio, 9),
        'favourites_count': favourites,
        'has_profile_image': 1,  # Genuine profiles usually have images
        'statuses_count': statuses,
        'engagement_score': engagement_score,
        'lang_label': random.choice([4, 5, 18, 14]),  # Common language codes
        'listed_count': listed
    }

def generate_fake_profile(profile_id):
    """Generate a fake-looking profile (suspicious metrics)"""
    # Fake profiles often have:
    # - Low followers, high following
    # - Few statuses
    # - Low or no description
    # - Recently created (reflected in low activity)
    
    followers = int(np.random.lognormal(2, 1.5))
    followers = int(np.clip(followers, 0, 500))
    
    # Fake profiles follow many people
    friends = int(np.random.lognormal(5, 1.5))
    friends = int(np.clip(friends, 100, 5000))
    
    # Low activity
    statuses = int(np.random.lognormal(2, 2))
    statuses = int(np.clip(statuses, 0, 100))
    
    favourites = int(np.random.lognormal(2, 2))
    favourites = int(np.clip(favourites, 0, 500))
    
    listed = int(np.random.exponential(0.5))
    listed = int(np.clip(listed, 0, 10))
    
    # Often no description or very short
    description_len = int(np.random.exponential(20))
    description_len = int(np.clip(description_len, 0, 100))
    
    follower_friend_ratio = followers / friends if friends > 0 else 0
    engagement_score = followers + friends + favourites + statuses
    
    utc_offsets = [0, 3600, 7200, -18000, -21600, -25200, -28800, 10800, 19800, 28800, 32400, 36000]
    
    return {
        'id': profile_id,
        'screen_name': generate_username(),
        'description_len': description_len,
        'followers_count': followers,
        'utc_offset': random.choice(utc_offsets),
        'friends_count': friends,
        'follower_friend_ratio': round(follower_friend_ratio, 9),
        'favourites_count': favourites,
        'has_profile_image': random.choice([0, 1]),  # Some fake profiles have no image
        'statuses_count': statuses,
        'engagement_score': engagement_score,
        'lang_label': random.choice([0, 4, 5, 14, 18, 20]),  # Various languages
        'listed_count': listed
    }

def generate_dataset(count=500, fake_ratio=0.5):
    """
    Generate a dataset with specified count of profiles
    
    Args:
        count: Total number of profiles to generate
        fake_ratio: Ratio of fake to genuine profiles (0.5 = 50% fake)
    """
    profiles = []
    num_fake = int(count * fake_ratio)
    num_genuine = count - num_fake
    
    print(f"Generating {count} profiles:")
    print(f"  - {num_genuine} genuine profiles ({(1-fake_ratio)*100:.0f}%)")
    print(f"  - {num_fake} fake profiles ({fake_ratio*100:.0f}%)")
    
    # Generate profiles
    profile_id = 1
    
    # Mix of fake and genuine
    profile_types = ['fake'] * num_fake + ['genuine'] * num_genuine
    random.shuffle(profile_types)
    
    for profile_type in profile_types:
        if profile_type == 'fake':
            profiles.append(generate_fake_profile(profile_id))
        else:
            profiles.append(generate_genuine_profile(profile_id))
        profile_id += 1
    
    return pd.DataFrame(profiles)

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic test data for fake profile detection')
    parser.add_argument('--count', type=int, default=500, help='Number of profiles to generate (default: 500)')
    parser.add_argument('--fake-ratio', type=float, default=0.5, help='Ratio of fake profiles (default: 0.5)')
    parser.add_argument('--output', type=str, default='data/test/test.csv', help='Output CSV file path')
    parser.add_argument('--preserve-original', action='store_true', help='Keep original data and append new rows')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("   Fake Profile Detection - Test Data Generator")
    print("="*70 + "\n")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if original file exists and should be preserved
    if args.preserve_original and output_path.exists():
        print(f"Loading existing data from {output_path}...")
        existing_df = pd.read_csv(output_path)
        print(f"  Found {len(existing_df)} existing profiles")
        
        # Generate new profiles starting from next ID
        start_id = existing_df['id'].max() + 1
        new_count = args.count - len(existing_df)
        
        if new_count <= 0:
            print(f"✅ Already have {len(existing_df)} profiles (target: {args.count})")
            return
        
        print(f"\nGenerating {new_count} additional profiles...")
        new_df = generate_dataset(new_count, args.fake_ratio)
        new_df['id'] = range(start_id, start_id + new_count)
        
        # Combine
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Generate fresh dataset
        df = generate_dataset(args.count, args.fake_ratio)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Successfully generated {len(df)} profiles")
    print(f"📁 Saved to: {output_path}")
    
    # Show statistics
    print("\n📊 Dataset Statistics:")
    print(f"  Total profiles: {len(df)}")
    print(f"  Average followers: {df['followers_count'].mean():.0f}")
    print(f"  Average friends: {df['friends_count'].mean():.0f}")
    print(f"  Average statuses: {df['statuses_count'].mean():.0f}")
    print(f"  Profiles with images: {df['has_profile_image'].sum()} ({df['has_profile_image'].mean()*100:.1f}%)")
    print(f"  Average follower/friend ratio: {df['follower_friend_ratio'].mean():.3f}")
    
    # Show sample
    print("\n📋 Sample profiles (first 5):")
    print(df[['id', 'screen_name', 'followers_count', 'friends_count', 'statuses_count']].head())
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()

