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
    
    def analyze_rainy_season_impact(self):
        """Analyze how rainy season affected traffic"""
        rain_keywords = [
            'lluvia', 'lluvias', 'lloviendo', 'llueve', 'temporal', 'aguacero',
            'inundacion', 'inundaciones', 'encharcamiento', 'agua', 'mojado',
            'humedo', 'precipitacion', 'tormenta', 'tormentas', 'chaparrón'
        ]
        
        # Create rain-related text filter
        def contains_rain_keywords(text):
            if pd.isna(text):
                return False
            text_lower = text.lower()
            return any(keyword in text_lower for keyword in rain_keywords)
        
        rain_tweets = self.df[self.df[self.text_column].apply(contains_rain_keywords)]
        
        print(f"\n🌧️ RAINY SEASON TRAFFIC IMPACT ANALYSIS")
        print(f"=" * 50)
        print(f"Rain-related tweets: {len(rain_tweets)} ({len(rain_tweets)/len(self.df)*100:.1f}%)")
        
        if len(rain_tweets) > 0:
            # Sentiment during rain
            if 'sentiment_label' in rain_tweets.columns:
                rain_sentiment = rain_tweets['sentiment_label'].value_counts()
                print(f"\nSentiment during rainy conditions:")
                for sentiment, count in rain_sentiment.items():
                    pct = count/len(rain_tweets)*100
                    print(f"  {sentiment}: {count} ({pct:.1f}%)")
            
            # Common phrases in rain tweets
            rain_text = ' '.join(rain_tweets[self.text_column].fillna(''))
            processed_rain_text = self.preprocess_text(rain_text)
            
            if processed_rain_text:
                from collections import Counter
                words = processed_rain_text.split()
                common_rain_words = Counter(words).most_common(10)
                print(f"\nMost common words in rain-related traffic tweets:")
                for word, count in common_rain_words:
                    print(f"  {word}: {count}")
        
        return rain_tweets
    
    def analyze_congested_areas(self):
        """Identify most congested areas mentioned in tweets"""
        # Guatemala City areas and zones
        guatemala_areas = [
            # Zones
            'zona 1', 'zona 2', 'zona 3', 'zona 4', 'zona 5', 'zona 6', 'zona 7', 
            'zona 8', 'zona 9', 'zona 10', 'zona 11', 'zona 12', 'zona 13', 
            'zona 14', 'zona 15', 'zona 16', 'zona 17', 'zona 18', 'zona 19', 'zona 21',
            
            # Major areas and neighborhoods
            'centro historico', 'centro', 'roosevelt', 'reforma', 'vista hermosa',
            'las americas', 'pradera', 'carretera al salvador', 'carretera interamericana',
            'mixco', 'villa nueva', 'petapa', 'santa catarina pinula', 'san jose pinula',
            'fraijanes', 'amatitlan', 'villa canales', 'chinautla', 'san pedro sacatepequez',
            
            # Major roads and highways
            'ca1', 'ca9', 'ruta al atlantico', 'ruta al pacifico', 'anillo periferico',
            'bulevar liberacion', 'calzada roosevelt', 'calzada san juan',
            '6a avenida', '7a avenida', 'avenida reforma', 'avenida las americas',
            'carretera a el salvador', 'autopista palín escuintla',
            
            # Commercial areas
            'centra norte', 'pradera concepcion', 'oakland mall', 'metronorte',
            'miraflores', 'portales', 'torre del reformador'
        ]
        
        # Traffic/congestion keywords
        traffic_keywords = [
            'trafico', 'tráfico', 'congestion', 'congestionamiento', 'tranque',
            'embotellamiento', 'atasco', 'lento', 'pesado', 'saturado',
            'bloqueado', 'cerrado', 'accidente', 'choque', 'colision'
        ]
        
        area_mentions = Counter()
        traffic_by_area = {}
        
        for idx, text in self.df[self.text_column].items():
            if pd.isna(text):
                continue
                
            text_lower = text.lower()
            
            # Check if it's a traffic-related tweet
            is_traffic_tweet = any(keyword in text_lower for keyword in traffic_keywords)
            
            if is_traffic_tweet:
                # Find mentioned areas
                for area in guatemala_areas:
                    if area in text_lower:
                        area_mentions[area] += 1
                        if area not in traffic_by_area:
                            traffic_by_area[area] = []
                        traffic_by_area[area].append(text)
        
        print(f"\n🚗 MOST CONGESTED AREAS ANALYSIS")
        print(f"=" * 50)
        
        if area_mentions:
            print(f"Top 15 most mentioned congested areas:")
            for area, count in area_mentions.most_common(15):
                print(f"  {area.title()}: {count} mentions")
                
            # Show sample tweets for top areas
            print(f"\nSample traffic reports for top areas:")
            for area, count in area_mentions.most_common(5):
                if area in traffic_by_area and traffic_by_area[area]:
                    print(f"\n{area.title()} ({count} mentions):")
                    sample_tweet = traffic_by_area[area][0][:100] + "..."
                    print(f"  Example: {sample_tweet}")
        else:
            print("No specific areas identified in traffic tweets")
            
        return area_mentions, traffic_by_area
    
    def analyze_temporal_patterns(self):
        """Analyze congestion patterns by time (requires datetime column)"""
        print(f"\n⏰ TEMPORAL CONGESTION PATTERNS")
        print(f"=" * 50)
        
        # Check if we have datetime information
        datetime_columns = [col for col in self.df.columns if any(term in col.lower() 
                           for term in ['time', 'date', 'created', 'timestamp', 'hora', 'fecha'])]
        
        if not datetime_columns:
            print("No datetime column found. Please ensure your dataframe has a datetime column.")
            print("Available columns:", list(self.df.columns))
            return None, None
            
        # Try to use the first datetime column
        dt_col = datetime_columns[0]
        print(f"Using datetime column: {dt_col}")
        
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(self.df[dt_col]):
                self.df[dt_col] = pd.to_datetime(self.df[dt_col])
            
            # Extract hour and day of week
            self.df['hour'] = self.df[dt_col].dt.hour
            self.df['day_of_week'] = self.df[dt_col].dt.day_name()
            
            # Filter traffic-related tweets
            traffic_keywords = ['trafico', 'tráfico', 'congestion', 'tranque', 'lento', 'pesado']
            traffic_tweets = self.df[self.df[self.text_column].str.contains(
                '|'.join(traffic_keywords), case=False, na=False)]
            
            if len(traffic_tweets) > 0:
                # Hourly patterns
                hourly_traffic = traffic_tweets['hour'].value_counts().sort_index()
                print(f"\nTraffic reports by hour:")
                for hour, count in hourly_traffic.items():
                    print(f"  {hour:02d}:00 - {count} reports")
                
                # Peak hours
                peak_hours = hourly_traffic.nlargest(3)
                print(f"\nPeak congestion hours:")
                for hour, count in peak_hours.items():
                    print(f"  {hour:02d}:00 - {count} reports")
                
                # Day of week patterns
                daily_traffic = traffic_tweets['day_of_week'].value_counts()
                print(f"\nTraffic reports by day of week:")
                for day, count in daily_traffic.items():
                    print(f"  {day}: {count} reports")
                    
                return hourly_traffic, daily_traffic
            else:
                print("No traffic-related tweets found for temporal analysis")
                return None, None
                
        except Exception as e:
            print(f"Error processing datetime column: {e}")
            return None, None
    
    def predict_congestion_patterns(self):
        """Analyze patterns to predict future congestion"""
        print(f"\n🔮 CONGESTION PREDICTION ANALYSIS")
        print(f"=" * 50)
        
        # Analyze historical patterns
        area_mentions, traffic_by_area = self.analyze_congested_areas()
        
        if area_mentions:
            # Identify consistently problematic areas
            top_areas = area_mentions.most_common(10)
            
            print(f"Areas likely to remain congested (based on frequency):")
            for area, count in top_areas:
                # Calculate likelihood based on frequency and sentiment
                traffic_texts = traffic_by_area.get(area, [])
                
                if len(traffic_texts) > 0 and 'sentiment_label' in self.df.columns:
                    # Check sentiment of tweets about this area
                    area_tweets = self.df[self.df[self.text_column].str.contains(
                        area, case=False, na=False)]
                    
                    if len(area_tweets) > 0:
                        negative_pct = (area_tweets['sentiment_label'] == 'negative').mean() * 100
                        likelihood = min(100, (count * 10) + negative_pct)
                        
                        status = "🔴 Very Likely" if likelihood > 70 else "🟡 Likely" if likelihood > 40 else "🟢 Possible"
                        print(f"  {area.title()}: {status} ({count} reports, {negative_pct:.0f}% negative sentiment)")
        
        # Seasonal patterns (if we have enough data)
        print(f"\nFactors suggesting continued congestion:")
        print(f"  • Consistent area mentions indicate structural traffic problems")
        print(f"  • High negative sentiment suggests ongoing frustration")
        print(f"  • Areas with infrastructure limitations likely to persist")
        
        return area_mentions.most_common(10) if area_mentions else []
    
    def generate_traffic_insights_report(self):
        """Generate comprehensive traffic-specific analysis report"""
        print("=" * 60)
        print("🚦 GUATEMALA CITY TRAFFIC ANALYSIS REPORT 🚦")
        print("=" * 60)
        
        # 1. Rainy season impact
        rain_tweets = self.analyze_rainy_season_impact()
        
        # 2. Congested areas
        area_mentions, traffic_by_area = self.analyze_congested_areas()
        
        # 3. Temporal patterns
        hourly_traffic, daily_traffic = self.analyze_temporal_patterns()
        
        # 4. Future predictions
        top_problematic_areas = self.predict_congestion_patterns()
        
        # Summary and recommendations
        print(f"\n📋 SUMMARY AND RECOMMENDATIONS")
        print(f"=" * 50)
        print(f"✅ Use this analysis to:")
        print(f"  • Identify peak congestion times for route planning")
        print(f"  • Focus infrastructure improvements on most mentioned areas")
        print(f"  • Prepare for seasonal challenges during rainy periods")
        print(f"  • Monitor sentiment trends to gauge public satisfaction")
    
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