import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Sentiment Analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans

# Additional libraries you might need to install:
# pip install textblob vaderSentiment wordcloud scikit-learn

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class SpanishTextAnalyzer:
    def __init__(self, df, text_column='text_clean_raw', hashtag_column='hastags'):
        self.df = df.copy()
        self.text_column = text_column
        self.hashtag_column = hashtag_column
        
        # Spanish stopwords
        self.spanish_stopwords = set(stopwords.words('spanish'))
        
        # Add custom Spanish stopwords for social media
        custom_stopwords = {
            'rt', 'via', 'http', 'https', 'www', 'com', 'co', 'gt', 
            'guatemala', 'trafico', 'si', 'no', 'ya', 'mas', 'muy',
            'ser', 'estar', 'tener', 'hacer', 'decir', 'ir', 'ver',
            'dar', 'saber', 'querer', 'llegar', 'pasar', 'deber'
        }
        self.spanish_stopwords.update(custom_stopwords)
        
        # Stemmer for Spanish
        self.stemmer = SnowballStemmer('spanish')
        
        # VADER analyzer (works reasonably well with Spanish)
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def preprocess_text(self, text):
        """Additional preprocessing for Spanish text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-záéíóúñü\s]', '', text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text, language='spanish')
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.spanish_stopwords and len(token) > 2]
        
        return ' '.join(tokens)
    
    def analyze_sentiment_textblob(self):
        """Sentiment analysis using TextBlob"""
        def get_sentiment(text):
            if pd.isna(text) or text == "":
                return 0, 'neutral'
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return polarity, 'positive'
            elif polarity < -0.1:
                return polarity, 'negative'
            else:
                return polarity, 'neutral'
        
        sentiments = self.df[self.text_column].apply(get_sentiment)
        self.df['sentiment_score'] = [s[0] for s in sentiments]
        self.df['sentiment_label'] = [s[1] for s in sentiments]
        
        return self.df
    
    def analyze_sentiment_vader(self):
        """Sentiment analysis using VADER"""
        def get_vader_sentiment(text):
            if pd.isna(text) or text == "":
                return 0, 'neutral'
            
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                return compound, 'positive'
            elif compound <= -0.05:
                return compound, 'negative'
            else:
                return compound, 'neutral'
        
        sentiments = self.df[self.text_column].apply(get_vader_sentiment)
        self.df['vader_score'] = [s[0] for s in sentiments]
        self.df['vader_label'] = [s[1] for s in sentiments]
        
        return self.df
    
    def analyze_hashtags(self):
        """Analyze hashtag patterns"""
        # Extract all hashtags
        all_hashtags = []
        for hashtag_list in self.df[self.hashtag_column]:
            try:
                # Check if the value is not null and not empty
                if pd.isna(hashtag_list):
                    continue
                
                # Handle different types of empty values
                if isinstance(hashtag_list, str):
                    if hashtag_list.strip() == '' or hashtag_list == '[]':
                        continue
                    # Try to evaluate string representation of list
                    try:
                        hashtags = eval(hashtag_list)
                    except:
                        # If eval fails, skip this entry
                        continue
                elif isinstance(hashtag_list, (list, tuple)):
                    hashtags = hashtag_list
                elif hasattr(hashtag_list, '__len__'):  # numpy array or similar
                    if len(hashtag_list) == 0:
                        continue
                    hashtags = hashtag_list
                else:
                    continue
                
                # Check if hashtags is actually a list/array with content
                if isinstance(hashtags, (list, tuple)) and len(hashtags) > 0:
                    all_hashtags.extend([str(tag).lower() for tag in hashtags if str(tag).strip() != ''])
                elif hasattr(hashtags, '__len__') and len(hashtags) > 0:
                    all_hashtags.extend([str(tag).lower() for tag in hashtags if str(tag).strip() != ''])
                    
            except Exception as e:
                # Skip problematic entries
                print(f"Warning: Skipped problematic hashtag entry: {hashtag_list} (Error: {e})")
                continue
        
        # Count hashtag frequency
        hashtag_counts = Counter(all_hashtags)
        
        return hashtag_counts
    
    def perform_topic_modeling(self, n_topics=5, method='lda'):
        """Perform topic modeling using LDA or NMF"""
        # Preprocess texts
        processed_texts = self.df[self.text_column].apply(self.preprocess_text)
        processed_texts = processed_texts[processed_texts != ""]
        
        if len(processed_texts) < 10:
            print("Not enough data for topic modeling")
            return None, None, None
        
        # Vectorize texts
        if method == 'lda':
            vectorizer = CountVectorizer(max_features=100, ngram_range=(1, 2))
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
            
            # LDA Topic Modeling
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(doc_term_matrix)
            
            model = lda
        else:  # NMF
            vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
            
            # NMF Topic Modeling
            nmf = NMF(n_components=n_topics, random_state=42)
            nmf.fit(doc_term_matrix)
            
            model = nmf
        
        return model, vectorizer, doc_term_matrix
    
    def display_topics(self, model, vectorizer, n_words=10):
        """Display top words for each topic"""
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        
        return topics
    
    def visualize_sentiment_distribution(self):
        """Create visualizations for sentiment analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # TextBlob sentiment distribution
        if 'sentiment_label' in self.df.columns:
            sentiment_counts = self.df['sentiment_label'].value_counts()
            axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('TextBlob Sentiment Distribution')
            
            axes[0, 1].hist(self.df['sentiment_score'], bins=30, alpha=0.7)
            axes[0, 1].set_title('TextBlob Sentiment Score Distribution')
            axes[0, 1].set_xlabel('Sentiment Score')
            axes[0, 1].set_ylabel('Frequency')
        
        # VADER sentiment distribution
        if 'vader_label' in self.df.columns:
            vader_counts = self.df['vader_label'].value_counts()
            axes[1, 0].pie(vader_counts.values, labels=vader_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('VADER Sentiment Distribution')
            
            axes[1, 1].hist(self.df['vader_score'], bins=30, alpha=0.7, color='orange')
            axes[1, 1].set_title('VADER Sentiment Score Distribution')
            axes[1, 1].set_xlabel('Sentiment Score')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def create_wordcloud(self):
        """Create word cloud from processed text"""
        processed_texts = self.df[self.text_column].apply(self.preprocess_text)
        all_text = ' '.join(processed_texts)
        
        if len(all_text) > 0:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud of Processed Text')
            plt.show()
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("=" * 50)
        print("SPANISH TEXT ANALYSIS REPORT")
        print("=" * 50)
        
        # Basic statistics
        print(f"\nDataset Overview:")
        print(f"Total tweets: {len(self.df)}")
        print(f"Non-empty texts: {self.df[self.text_column].notna().sum()}")
        
        # Sentiment Analysis Results
        if 'sentiment_label' in self.df.columns:
            print(f"\nTextBlob Sentiment Analysis:")
            print(self.df['sentiment_label'].value_counts())
            print(f"Average sentiment score: {self.df['sentiment_score'].mean():.3f}")
        
        if 'vader_label' in self.df.columns:
            print(f"\nVADER Sentiment Analysis:")
            print(self.df['vader_label'].value_counts())
            print(f"Average VADER score: {self.df['vader_score'].mean():.3f}")
        
        # Hashtag Analysis
        hashtag_counts = self.analyze_hashtags()
        if hashtag_counts:
            print(f"\nTop 10 Hashtags:")
            for hashtag, count in hashtag_counts.most_common(10):
                print(f"#{hashtag}: {count}")

# Example usage:
"""
# Assuming your dataframe is called 'df'
analyzer = SpanishTextAnalyzer(df, 'text_clean_raw', 'hastags')

# Perform sentiment analysis
df_with_sentiment = analyzer.analyze_sentiment_textblob()
df_with_sentiment = analyzer.analyze_sentiment_vader()

# Perform topic modeling
lda_model, vectorizer, doc_term_matrix = analyzer.perform_topic_modeling(n_topics=5, method='lda')
if lda_model:
    topics = analyzer.display_topics(lda_model, vectorizer)

# Create visualizations
analyzer.visualize_sentiment_distribution()
analyzer.create_wordcloud()

# Generate comprehensive report
analyzer.generate_report()

# Access the results
print("Sample results:")
print(df_with_sentiment[['text_clean_raw', 'sentiment_label', 'vader_label', 'sentiment_score', 'vader_score']].head())
"""