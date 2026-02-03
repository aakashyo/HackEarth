import pandas as pd
import numpy as np
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import os

class EmotionDataLoader:
    PRIMARY_GROUPS = {
        'Sadness': ['Longing', 'Melancholy', 'Sadness', 'Grief', 'Sorrow', 'Loss', 'Regret', 'Nostalgia', 'Despair', 'Heartbreak'],
        'Joy': ['Love', 'Happiness', 'Joy', 'Contentment', 'Bliss', 'Euphoria', 'Delight', 'Pleasure', 'Gratitude', 'Hope'],
        'Fear': ['Fear', 'Caution', 'Anxiety', 'Dread', 'Terror', 'Worry', 'Apprehension', 'Nervousness'],
        'Contemplation': ['Philosophy', 'Wisdom', 'Reflection', 'Introspection', 'Meditation', 'Spirituality', 'Enlightenment', 'Transcendence'],
        'Pride': ['Patriotism', 'Pride', 'Honor', 'Dignity', 'Achievement', 'Triumph', 'Victory', 'Glory'],
        'Nature': ['Nature', 'Beauty', 'Serenity', 'Peace', 'Tranquility', 'Harmony', 'Wonder', 'Awe'],
        'Anger': ['Anger', 'Frustration', 'Resentment', 'Rage', 'Indignation', 'Bitterness', 'Jealousy', 'Envy'],
        'Devotion': ['Devotion', 'Faith', 'Reverence', 'Worship', 'Piety', 'Respect', 'Admiration', 'Loyalty']
    }
    
    SECONDARY_GROUPS = {
        'Sadness': ['Longing', 'Melancholy', 'Sadness', 'Grief', 'Sorrow', 'Loss', 'Regret', 'Nostalgia', 'Despair', 'Heartbreak'],
        'Joy': ['Love', 'Happiness', 'Joy', 'Contentment', 'Bliss', 'Euphoria', 'Delight', 'Pleasure', 'Gratitude', 'Hope'],
        'Fear': ['Fear', 'Caution', 'Anxiety', 'Dread', 'Terror', 'Worry', 'Apprehension', 'Nervousness'],
        'Contemplation': ['Philosophy', 'Wisdom', 'Reflection', 'Introspection', 'Meditation', 'Spirituality', 'Enlightenment', 'Transcendence'],
        'Pride': ['Patriotism', 'Pride', 'Honor', 'Dignity', 'Achievement', 'Triumph', 'Victory', 'Glory'],
        'Nature': ['Nature', 'Beauty', 'Serenity', 'Peace', 'Tranquility', 'Harmony', 'Wonder', 'Awe'],
        'Anger': ['Anger', 'Frustration', 'Resentment', 'Rage', 'Indignation', 'Bitterness', 'Jealousy', 'Envy'],
        'Devotion': ['Devotion', 'Faith', 'Reverence', 'Worship', 'Piety', 'Respect', 'Admiration', 'Loyalty']
    }
    
    def __init__(self, primary_path, secondary_path, use_grouping=True):
        self.primary_path = primary_path
        self.secondary_path = secondary_path
        self.use_grouping = use_grouping
        self.primary_labels = []
        self.secondary_labels = []
        self.primary_label2id = {}
        self.secondary_label2id = {}
        self.primary_id2label = {}
        self.secondary_id2label = {}
        
        if use_grouping:
            self.primary_mapping = {}
            for group, emotions in self.PRIMARY_GROUPS.items():
                for e in emotions:
                    self.primary_mapping[e] = group
            self.secondary_mapping = {}
            for group, emotions in self.SECONDARY_GROUPS.items():
                for e in emotions:
                    self.secondary_mapping[e] = group
        
    def normalize_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = unicodedata.normalize('NFC', text)
        text = ' '.join(text.split())
        return text
    
    def map_emotion(self, emotion, mapping):
        if emotion in mapping:
            return mapping[emotion]
        return 'Other'
    
    def load_data(self):
        primary_df = pd.read_csv(self.primary_path)
        secondary_df = pd.read_csv(self.secondary_path)
        
        primary_df.columns = primary_df.columns.str.strip()
        secondary_df.columns = secondary_df.columns.str.strip()
        
        primary_df['Poem'] = primary_df['Poem'].apply(self.normalize_text)
        secondary_df['Poem'] = secondary_df['Poem'].apply(self.normalize_text)
        
        merged_df = pd.merge(
            primary_df[['Poem', 'Source', 'Primary']],
            secondary_df[['Poem', 'Secondary']],
            on='Poem',
            how='inner'
        )
        
        merged_df = merged_df.dropna(subset=['Primary', 'Secondary'])
        merged_df = merged_df[merged_df['Poem'].str.len() > 0]
        
        if self.use_grouping:
            merged_df['Primary'] = merged_df['Primary'].apply(lambda x: self.map_emotion(x, self.primary_mapping))
            merged_df['Secondary'] = merged_df['Secondary'].apply(lambda x: self.map_emotion(x, self.secondary_mapping))
        
        self.primary_labels = sorted(merged_df['Primary'].unique().tolist())
        self.secondary_labels = sorted(merged_df['Secondary'].unique().tolist())
        
        self.primary_label2id = {label: idx for idx, label in enumerate(self.primary_labels)}
        self.secondary_label2id = {label: idx for idx, label in enumerate(self.secondary_labels)}
        self.primary_id2label = {idx: label for label, idx in self.primary_label2id.items()}
        self.secondary_id2label = {idx: label for label, idx in self.secondary_label2id.items()}
        
        merged_df['primary_id'] = merged_df['Primary'].map(self.primary_label2id)
        merged_df['secondary_id'] = merged_df['Secondary'].map(self.secondary_label2id)
        
        return merged_df
    
    def get_train_test_split(self, df, test_size=0.2, random_state=42):
        df['stratify_key'] = df['Primary'] + '_' + df['Secondary']
        
        value_counts = df['stratify_key'].value_counts()
        valid_keys = value_counts[value_counts >= 2].index
        df_stratifiable = df[df['stratify_key'].isin(valid_keys)]
        df_non_stratifiable = df[~df['stratify_key'].isin(valid_keys)]
        
        if len(df_stratifiable) > 0:
            train_strat, test_strat = train_test_split(
                df_stratifiable,
                test_size=test_size,
                random_state=random_state,
                stratify=df_stratifiable['stratify_key']
            )
            train_df = pd.concat([train_strat, df_non_stratifiable])
            test_df = test_strat
        else:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state
            )
        
        train_df = train_df.drop('stratify_key', axis=1)
        test_df = test_df.drop('stratify_key', axis=1)
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def compute_class_weights(self, df):
        primary_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(range(len(self.primary_labels))),
            y=df['primary_id'].values
        )
        
        secondary_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(range(len(self.secondary_labels))),
            y=df['secondary_id'].values
        )
        
        return primary_weights, secondary_weights
    
    def get_label_info(self):
        return {
            'num_primary': len(self.primary_labels),
            'num_secondary': len(self.secondary_labels),
            'primary_labels': self.primary_labels,
            'secondary_labels': self.secondary_labels,
            'primary_label2id': self.primary_label2id,
            'secondary_label2id': self.secondary_label2id,
            'primary_id2label': self.primary_id2label,
            'secondary_id2label': self.secondary_id2label
        }


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = "C:\\Users\\akash\\Downloads\\HACK"
    
    loader = EmotionDataLoader(
        primary_path=os.path.join(data_path, "dataset.csv.xlsx - Sheet1.csv"),
        secondary_path=os.path.join(data_path, "Secondary_Emotions.xlsx - Sheet1.csv")
    )
    
    df = loader.load_data()
    print(f"Total samples: {len(df)}")
    print(f"Primary emotions: {len(loader.primary_labels)}")
    print(f"Secondary emotions: {len(loader.secondary_labels)}")
    
    train_df, test_df = loader.get_train_test_split(df)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    primary_weights, secondary_weights = loader.compute_class_weights(train_df)
    print(f"Primary class weights shape: {primary_weights.shape}")
    print(f"Secondary class weights shape: {secondary_weights.shape}")
