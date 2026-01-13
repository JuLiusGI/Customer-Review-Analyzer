import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_reviews(num_reviews=100):
    topics = {
        'quality': [
            "The product quality is amazing.",
            "Material feels very cheap.",
            "Built to last, very sturdy.",
            "Broke after two days of use.",
            "Excellent craftsmanship.",
            "Not what I expected, poor finish."
        ],
        'shipping': [
            "Arrived earlier than expected!",
            "Shipping took forever.",
            "Package was damaged upon arrival.",
            "Fast delivery and well packaged.",
            "Tracking number didn't work.",
            "Delivered to the wrong address."
        ],
        'service': [
            "Customer service was very helpful.",
            "Rude support staff, avoided my questions.",
            "Quick response to my inquiry.",
            "No one replied to my emails.",
            "Great experience with the support team.",
            "Worst customer service ever."
        ]
    }

    reviews = []
    start_date = datetime.now() - timedelta(days=365)

    for i in range(num_reviews):
        topic = random.choice(list(topics.keys()))
        base_text = random.choice(topics[topic])
        
        # Add some variation
        variations = [
            " Highly recommended.",
            " Would not buy again.",
            " Pretty decent overall.",
            " I'm very satisfied.",
            " Extremely disappointed.",
            ""
        ]
        text = base_text + random.choice(variations)
        
        # Assign rating based on sentiment roughly (simplified for dummy data)
        if any(w in text.lower() for w in ['amazing', 'excellent', 'fast', 'helpful', 'great', 'satisfied', 'recommended']):
            rating = random.randint(4, 5)
        elif any(w in text.lower() for w in ['cheap', 'broke', 'poor', 'forever', 'damaged', 'rude', 'worst', 'disappointed']):
            rating = random.randint(1, 2)
        else:
            rating = 3

        timestamp = start_date + timedelta(days=random.randint(0, 365))
        
        reviews.append({
            'id': i + 1,
            'text': text,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'rating': rating,
            'topic': topic # Ground truth for checking clustering later if needed, though real app won't have this
        })

    return pd.DataFrame(reviews)

if __name__ == "__main__":
    print("Generating dummy reviews...")
    df = generate_reviews(150)
    output_path = "data/reviews.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} reviews to {output_path}")
    print(df.head())
