from typing import Any, Dict
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import schedule
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import psutil  # For system monitoring
import platform
from collections import defaultdict
import logging
import warnings
import os
import random
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
WEATHER_API_KEY = "151cdf27c929cdb9d2213eabbcdadef9"
class EnhancedUserActivityTracker:
    def __init__(self):
        self.user_sessions = {}
        self.device_data = {}
        self.browsing_history = defaultdict(list)
        self.system_metrics = {}

        # Load activity dataset
        # self.activity_data = pd.read_csv(activity_dataset_path)
        
    def track_system_metrics(self, user_id):
        """Track system-level metrics"""
        metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'battery': psutil.sensors_battery().percent if hasattr(psutil, 'sensors_battery') and 
                      psutil.sensors_battery() else None,
            'system': platform.system(),
            'browser': self.device_data.get(user_id, {}).get('browser', 'unknown')
        }
        self.system_metrics[user_id] = metrics
        return metrics

    def track_browsing_behavior(self, user_id, page_data):
        """Track detailed browsing behavior"""
        current_time = datetime.now()
        browsing_data = {
            'timestamp': current_time,
            'page_url': page_data.get('url'),
            'time_spent': page_data.get('time_spent', 0),
            'scroll_depth': page_data.get('scroll_depth', 0),
            'clicks': page_data.get('clicks', 0),
            'mouse_movement': page_data.get('mouse_movement', 0)
        }
        self.browsing_history[user_id].append(browsing_data)

    def track_activity(self, user_id, activity_type, details=None):
        """Enhanced activity tracking"""
        current_time = datetime.now()
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'session_start': current_time,
                'last_activity': current_time,
                'activity_count': 0,
                'activities': defaultdict(list),
                'product_views': defaultdict(int),
                'cart_items': [],
                'search_history': [],
                'category_interests': defaultdict(int),
                'device_info': self.track_system_metrics(user_id)
            }
        
        session = self.user_sessions[user_id]
        session['last_activity'] = current_time
        session['activity_count'] += 1
        
        # Track specific activity with details
        activity_data = {
            'timestamp': current_time,
            'details': details
        }
        session['activities'][activity_type].append(activity_data)
        
        # Update specific metrics based on activity type
        if activity_type == 'product_view' and details:
            session['product_views'][details['product_id']] += 1
            if 'category' in details:
                session['category_interests'][details['category']] += 1
        elif activity_type == 'search' and details:
            session['search_history'].append(details['query'])
        elif activity_type == 'add_to_cart' and details:
            session['cart_items'].append(details['product_id'])
        elif activity_type == 'browsing' and details:
            self.track_browsing_behavior(user_id, details)
            
    def get_user_activity(self, user_id):
        """Get comprehensive user activity data"""
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            current_time = datetime.now()
            duration = (current_time - session['session_start']).total_seconds() / 60
            
            # Get recent browsing history
            recent_browsing = self.browsing_history[user_id][-10:] if user_id in self.browsing_history else []
            
            return {
                'session_duration': duration,
                'activity_count': session['activity_count'],
                'product_views': dict(session['product_views']),
                'cart_items': len(session['cart_items']),
                'search_history': session['search_history'][-5:],
                'category_interests': dict(session['category_interests']),
                'recent_browsing': [
                    {**browsing_data, 'timestamp': browsing_data['timestamp'].isoformat()}
                    for browsing_data in recent_browsing
                ],
                'device_info': self.track_system_metrics(user_id),
                'last_activity': session['last_activity'].isoformat() 
            }
        return None

class WeatherAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.weather_cache = {}
        self.cache_duration = timedelta(minutes=30)
        
    def get_detailed_weather(self, city):
        """Get detailed weather information with caching"""
        current_time = datetime.now()
        
        # Check cache
        if city in self.weather_cache:
            cache_time, cached_data = self.weather_cache[city]
            if current_time - cache_time < self.cache_duration:
                return cached_data
        
        # Fetch new weather data
        try:
            base_url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                weather_data = response.json()

                coordinates = {
                    'latitude': weather_data['coord']['lat'],
                    'longitude': weather_data['coord']['lon']
                }
                
                # Process weather conditions
                weather_info = {
                    'temperature': weather_data['main']['temp'],
                    'humidity': weather_data['main']['humidity'],
                    'wind_speed': weather_data['wind']['speed'],
                    'condition': weather_data['weather'][0]['main'].lower(),
                    'description': weather_data['weather'][0]['description'],
                    'pressure': weather_data['main']['pressure'],
                    'clouds': weather_data.get('clouds', {}).get('all', 0),
                    'rain': weather_data.get('rain', {}).get('1h', 0),
                    'time_of_day': 'day' if 6 <= datetime.now().hour <= 18 else 'night',
                    'coordinates': coordinates 
                }
                
                # Cache the processed data
                self.weather_cache[city] = (current_time, weather_info)
                return weather_info
                
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None
class ContextAwarenessEngine:
    def __init__(self):
        self.seasonal_data = self._initialize_seasonal_data()
        self.regional_preferences = {}
        self.events_calendar = self._initialize_events_calendar()
        self.last_update = datetime.now()
        
    def _initialize_seasonal_data(self):
        """Initialize seasonal trend data"""
        seasons = {
            # Northern Hemisphere seasons
            'spring_north': {'months': [3, 4, 5], 'trends': ['gardening', 'outdoor', 'light clothing']},
            'summer_north': {'months': [6, 7, 8], 'trends': ['beach', 'travel', 'cooling']},
            'fall_north': {'months': [9, 10, 11], 'trends': ['back to school', 'harvest', 'light jacket']},
            'winter_north': {'months': [12, 1, 2], 'trends': ['holiday', 'winter gear', 'indoor']},
            
            # Southern Hemisphere seasons (inverted)
            'spring_south': {'months': [9, 10, 11], 'trends': ['gardening', 'outdoor', 'light clothing']},
            'summer_south': {'months': [12, 1, 2], 'trends': ['beach', 'travel', 'cooling']},
            'fall_south': {'months': [3, 4, 5], 'trends': ['back to school', 'harvest', 'light jacket']},
            'winter_south': {'months': [6, 7, 8], 'trends': ['winter gear', 'indoor activities']}
        }
        return seasons
    
    def _initialize_events_calendar(self):
        """Initialize calendar of major events"""
        events = {
            # Format: 'MM-DD': {'name': 'Event Name', 'trends': ['keyword1', 'keyword2']}
            '01-01': {'name': 'New Year', 'trends': ['resolution', 'fitness', 'organization']},
            '02-14': {'name': 'Valentine\'s Day', 'trends': ['gift', 'romantic', 'chocolate']},
            '10-31': {'name': 'Halloween', 'trends': ['costume', 'candy', 'decoration']},
            '11-01': {'name': 'Diwali', 'trends': ['light', 'gift', 'decoration']},
            '12-25': {'name': 'Christmas', 'trends': ['gift', 'decoration', 'winter']}
            # Add more events as needed
        }
        return events
    
    def load_regional_preferences(self, file_path=None):
        """Load regional preference data from file or initialize with defaults"""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    self.regional_preferences = json.load(f)
            except Exception as e:
                logger.error(f"Error loading regional preferences: {e}")
                self._initialize_default_regional_prefs()
        else:
            self._initialize_default_regional_prefs()
    
    def _initialize_default_regional_prefs(self):
        """Initialize default regional preference data"""
        self.regional_preferences = {
            'North America': ['electronics', 'home appliances', 'outdoor gear'],
            'Europe': ['fashion', 'luxury goods', 'eco-friendly'],
            'Asia': ['electronics', 'beauty products', 'traditional items'],
            'South America': ['colorful fashion', 'local crafts', 'fresh foods'],
            'Africa': ['mobile technology', 'traditional clothing', 'practical items'],
            'Australia': ['outdoor', 'beach gear', 'barbecue items'],
            'India': ['traditional wear', 'electronics', 'jewelry']
        }
    
    def get_current_season(self, latitude=None):
        """Determine current season based on month and hemisphere"""
        current_month = datetime.now().month
        
        # Determine hemisphere (North by default)
        hemisphere = 'south' if latitude and latitude < 0 else 'north'
        
        for season, data in self.seasonal_data.items():
            if hemisphere in season and current_month in data['months']:
                return {'season': season, 'trends': data['trends']}
        
        return {'season': 'unknown', 'trends': []}
    
    def get_upcoming_events(self, days_ahead=7):
        """Get upcoming events within specified days"""
        current_date = datetime.now()
        upcoming = []
        
        for day_offset in range(days_ahead):
            check_date = current_date + timedelta(days=day_offset)
            date_key = check_date.strftime('%m-%d')
            
            if date_key in self.events_calendar:
                event = self.events_calendar[date_key]
                event['days_away'] = day_offset
                upcoming.append(event)
        
        return upcoming
    
    def map_location_to_region(self, city, country=None):
        """Map a city to a broader region for regional preferences"""
        # This would ideally use a geolocation API or database
        # For now, using a simple mapping for demonstration
        region_map = {
            'New York': 'North America',
            'London': 'Europe',
            'Tokyo': 'Asia',
            'Sao Paulo': 'South America',
            'Cairo': 'Africa',
            'Sydney': 'Australia',
            'Mumbai': 'India',
            'Kolkata': 'India',
            'Delhi': 'India'
            # Add more cities as needed
        }
        
        return region_map.get(city, 'Global')
    
    def get_context_factors(self, city, latitude=None):
        """Get all contextual factors for given location"""
        # Get seasonal information
        season_info = self.get_current_season(latitude)
        
        # Get regional preferences
        region = self.map_location_to_region(city)
        regional_prefs = self.regional_preferences.get(region, [])
        
        # Get upcoming events
        events = self.get_upcoming_events()
        
        return {
            'season': season_info,
            'region': region,
            'regional_preferences': regional_prefs,
            'upcoming_events': events
        }
class MultimodalSentimentAnalyzer:
    def __init__(self):
        # Initialize NLTK sentiment analyzer for text
        import nltk
        nltk.download('vader_lexicon')
        self.text_analyzer = SentimentIntensityAnalyzer()
        self.sentiment_cache = {}
        
        # Feature flags for different modalities
        self.image_analysis_enabled = False
        self.voice_analysis_enabled = False
        
        # Track sentiment history
        self.sentiment_history = defaultdict(list)

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def enable_image_analysis(self, api_key=None):
        """Enable image analysis with optional API key"""
        if api_key:
            self.image_api_key = api_key
            self.image_analysis_enabled = True
            logger.info("Image sentiment analysis enabled")
        else:
            logger.warning("No API key provided for image analysis")

    def extract_labels_from_image(self, image_path, candidate_labels=None):
        """Extract top matching labels from the uploaded image using CLIP"""
        if not candidate_labels:
            candidate_labels = [
                "shirt", "jeans", "shoes", "dress", "phone", "laptop", "bag", 
                "headphones", "watch", "t-shirt", "jacket", "sunglasses", "toy"
            ]

        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # similarity scores
        probs = logits_per_image.softmax(dim=1).detach().numpy()[0]

        label_scores = list(zip(candidate_labels, probs))
        label_scores.sort(key=lambda x: x[1], reverse=True)

        top_labels = [label for label, score in label_scores[:3]]  # Top 3 labels
        return top_labels
    
    def enable_voice_analysis(self, api_key=None):
        """Enable voice analysis with optional API key"""
        if api_key:
            self.voice_api_key = api_key
            self.voice_analysis_enabled = True
            logger.info("Voice sentiment analysis enabled")
        else:
            logger.warning("No API key provided for voice analysis")
    
    def analyze_text(self, text):
        """Analyze sentiment from text"""
        if not text:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
            
        # Check cache first
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]
            
        sentiment = self.text_analyzer.polarity_scores(text)
        self.sentiment_cache[text] = sentiment
        return sentiment
    
    def _mock_image_analysis(self, image_path):
        """Mock implementation of image sentiment analysis"""
        return {
            'joy': round(random.random(), 2),
            'sadness': round(random.random() * 0.5, 2),
            'excitement': round(random.random(), 2),
            'interest': round(random.random(), 2)
        }
    
    def analyze_image(self, image_path):
        """Analyze sentiment from image"""
        if not self.image_analysis_enabled:
            logger.warning("Image analysis not enabled")
            return None
            
        try:
            # This would call an actual API in production
            return self._mock_image_analysis(image_path)
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return None
    
    def _mock_voice_analysis(self, audio_path):
        """Mock implementation of voice sentiment analysis"""
        # In a real implementation, this would call a speech analysis API
        return {
            'tone': random.choice(['neutral', 'happy', 'excited', 'calm', 'concerned']),
            'confidence': round(random.random() * 0.5 + 0.5, 2),
            'pitch_variation': round(random.random(), 2)
        }
    
    def analyze_voice(self, audio_path):
        """Analyze sentiment from voice recording"""
        if not self.voice_analysis_enabled:
            logger.warning("Voice analysis not enabled")
            return None
            
        try:
            # This would call an actual API in production
            return self._mock_voice_analysis(audio_path)
        except Exception as e:
            logger.error(f"Error in voice analysis: {e}")
            return None
    
    def analyze_multimodal(self, user_id, text=None, image_path=None, audio_path=None):
        """Perform multimodal sentiment analysis"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'modalities_used': []
        }
        
        # Analyze text if provided
        if text:
            results['text_sentiment'] = self.analyze_text(text)
            results['modalities_used'].append('text')
        
        # Analyze image if provided and enabled
        if image_path and self.image_analysis_enabled:
            results['image_sentiment'] = self.analyze_image(image_path)
            results['modalities_used'].append('image')
        
        # Analyze voice if provided and enabled
        if audio_path and self.voice_analysis_enabled:
            results['voice_sentiment'] = self.analyze_voice(audio_path)
            results['modalities_used'].append('voice')
        
        # Calculate combined sentiment score
        if results['modalities_used']:
            combined_score = 0
            
            if 'text_sentiment' in results:
                combined_score += results['text_sentiment']['compound']
                
            if 'image_sentiment' in results:
                img_sentiment = results['image_sentiment']
                img_score = img_sentiment['joy'] - img_sentiment['sadness']
                combined_score += img_score
                
            if 'voice_sentiment' in results:
                voice_sentiment = results['voice_sentiment']
                voice_map = {'happy': 0.8, 'excited': 0.9, 'calm': 0.3, 'neutral': 0, 'concerned': -0.3}
                voice_score = voice_map.get(voice_sentiment['tone'], 0) * voice_sentiment['confidence']
                combined_score += voice_score
                
            # Normalize combined score
            num_modalities = len(results['modalities_used'])
            results['combined_sentiment'] = combined_score / num_modalities
            
            # Store in history
            self.sentiment_history[user_id].append({
                'timestamp': results['timestamp'],
                'combined_sentiment': results['combined_sentiment'],
                'modalities': results['modalities_used']
            })
            
            # Keep history to a reasonable size
            if len(self.sentiment_history[user_id]) > 100:
                self.sentiment_history[user_id] = self.sentiment_history[user_id][-100:]
        
        return results
    
    def get_sentiment_trends(self, user_id, days=7):
        """Get sentiment trends over time for a user"""
        if user_id not in self.sentiment_history:
            return None
            
        history = self.sentiment_history[user_id]
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=days)
        
        # Filter history to specified time period
        recent_history = [
            item for item in history 
            if datetime.fromisoformat(item['timestamp']) >= cutoff_time
        ]
        
        if not recent_history:
            return None
            
        # Calculate average and trend
        sentiment_values = [item['combined_sentiment'] for item in recent_history]
        avg_sentiment = sum(sentiment_values) / len(sentiment_values)
        
        # Simple trend calculation
        if len(sentiment_values) >= 2:
            first_half = sentiment_values[:len(sentiment_values)//2]
            second_half = sentiment_values[len(sentiment_values)//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            trend = 'increasing' if second_avg > first_avg else 'decreasing' if second_avg < first_avg else 'stable'
        else:
            trend = 'insufficient data'
        
        return {
            'average_sentiment': avg_sentiment,
            'trend': trend,
            'data_points': len(recent_history),
            'period_days': days
        }
class EnhancedRecommendationSystem:
    def __init__(self, weather_api_key):
        self.product_data = None
        self.user_tracker = EnhancedUserActivityTracker()
        self.weather_analyzer = WeatherAnalyzer(weather_api_key)

        self.context_engine = ContextAwarenessEngine()
        self.sentiment_analyzer = MultimodalSentimentAnalyzer()

        # Initialize NLTK for text sentiment
        self.sia = SentimentIntensityAnalyzer()
        self.sentiment_cache = {}
        
        # Weather rules
        self.weather_rules = {
            'rainy': {
                'items': ['umbrella', 'raincoat', 'boots', 'waterproof'],
                'categories': ['rain gear', 'indoor activities']
            },
            'cold': {
                'items': ['jacket', 'sweater', 'winter', 'warm'],
                'categories': ['winter wear', 'hot beverages']
            },
            'hot': {
                'items': ['summer', 'sunny', 'cooling', 'beach'],
                'categories': ['summer wear', 'cooling products']
            },
            'moderate': {
                'items': ['casual', 'everyday', 'comfortable'],
                'categories': ['casual wear', 'accessories']
            },
            'stormy': {
                'items': ['emergency', 'waterproof', 'safety'],
                'categories': ['emergency supplies', 'indoor equipment']
            }
        }

        # Initialize context awareness data
        self.context_engine.load_regional_preferences()

    

    def get_weather_based_products(self, weather_info):
        """Get products based on detailed weather conditions"""
        condition = weather_info['condition'].lower()
        temperature = weather_info['temperature']
        
        # Determine weather category
        if 'rain' in condition or 'drizzle' in condition:
            category = 'rainy'
        elif 'storm' in condition or 'thunder' in condition:
            category = 'stormy'
        elif temperature < 15:
            category = 'cold'
        elif temperature > 25:
            category = 'hot'
        else:
            category = 'moderate'
            
        # Get relevant products based on weather category
        weather_rules = self.weather_rules[category]
        relevant_products = self.product_data[
            self.product_data['product_category_tree'].str.contains('|'.join(weather_rules['items']), 
            case=False, na=False) |
            self.product_data['product_category_tree'].str.contains('|'.join(weather_rules['categories']), 
            case=False, na=False)
        ]
        
        return relevant_products, category
    
    def preprocess_voice_for_ml(self, voice_text: str) -> str:
        """Preprocess voice text for better ML understanding"""
        if not voice_text:
            return ""
        
        # Convert to lowercase and clean
        processed = voice_text.lower().strip()
        
        # Extract key product-related keywords
        product_keywords = []
        weather_keywords = []
        
        # Check for weather-related terms
        weather_terms = ['rain', 'sunny', 'cold', 'hot', 'winter', 'summer', 'monsoon']
        for term in weather_terms:
            if term in processed:
                weather_keywords.append(term)
        
        # Check for product-related terms
        product_terms = ['buy', 'need', 'want', 'looking for', 'search', 'find']
        for term in product_terms:
            if term in processed:
                product_keywords.append(term)
        
        return processed

    def generate_voice_based_recommendations(self, voice_text: str, user_id: str = "voice_user", city: str = "Kolkata") -> Dict[str, Any]:
        """Generate recommendations based on voice input using ML system"""
        if not voice_text or not voice_text.strip():
            return {"success": False, "error": "No voice text provided"}
        
        try:
            # Process voice text for better ML understanding
            processed_query = self.preprocess_voice_for_ml(voice_text)
            
            # Analyze sentiment from voice text
            text_sentiment = self.sentiment_analyzer.analyze_text(voice_text)
            
            # Track this voice interaction as user activity
            self.user_tracker.track_activity(user_id, 'voice_search', {
                'query': voice_text,
                'processed_query': processed_query,
                'timestamp': datetime.now().isoformat()
            })
            
            # Generate enhanced recommendations with voice context
            enhanced_results = self.get_enhanced_recommendations(
                user_id=user_id,
                city=city,
                user_text=voice_text  # Use voice text as text input for sentiment analysis
            )
            
            if not enhanced_results:
                return {"success": False, "error": "Could not generate recommendations"}
            
            # Extract relevant products based on voice keywords
            voice_filtered_recommendations = self.filter_by_voice_keywords(
                enhanced_results['recommendations'], 
                processed_query
            )
            
            # Analyze the voice interaction with multimodal system
            multimodal_analysis = self.sentiment_analyzer.analyze_multimodal(
                user_id=user_id,
                text=voice_text
            )

            return {
                "success": True,
                "recommendations": voice_filtered_recommendations.to_dict('records') if not voice_filtered_recommendations.empty else [],
                "processed_query": processed_query,
                "original_voice_text": voice_text,
                "sentiment_analysis": text_sentiment,
                "multimodal_analysis": multimodal_analysis,
                "weather_info": enhanced_results.get('weather'),
                "context_factors": enhanced_results.get('context_factors'),
                "total_recommendations": len(voice_filtered_recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error in voice-ML integration: {e}")
            return {"success": False, "error": str(e)}

    def filter_by_voice_keywords(self, recommendations_df, voice_query):
        """Filter recommendations based on voice keywords"""
        if recommendations_df.empty or not voice_query:
            return recommendations_df.head(10)
    
        # Extract keywords from voice query
        keywords = voice_query.lower().split()
    
    # Create relevance score based on keyword matches
        recommendations_df = recommendations_df.copy()
        recommendations_df['voice_relevance'] = 0
    
        for keyword in keywords:
            if keyword in ['find', 'search', 'get', 'show', 'want', 'need', 'buy']:
                continue  # Skip action words
            
        # Check matches in product name and category
            name_matches = recommendations_df['product_name'].str.contains(
            keyword, case=False, na=False
            )
            category_matches = recommendations_df['product_category_tree'].str.contains(
            keyword, case=False, na=False
            )
        
        # Increase relevance score for matches
            recommendations_df.loc[name_matches, 'voice_relevance'] += 2
            recommendations_df.loc[category_matches, 'voice_relevance'] += 1
    
    # If we have relevance matches, prioritize them
        if recommendations_df['voice_relevance'].max() > 0:
            recommendations_df = recommendations_df.sort_values(
            ['voice_relevance', 'final_score'], 
            ascending=[False, False]
        )
    
    # Return top results
        return recommendations_df.head(15)

    def load_data(self, file_path):
        """Load and preprocess product data with ratings"""
        try:
            self.product_data = pd.read_csv(file_path)
            
            # Convert ratings to numeric, handle any non-numeric values
            self.product_data['product_rating'] = pd.to_numeric(
                self.product_data['product_rating'], 
                errors='coerce'
            ).fillna(0)
            
            self.product_data['overall_rating'] = pd.to_numeric(
                self.product_data['overall_rating'], 
                errors='coerce'
            ).fillna(0)
            
            logger.info(f"Loaded {len(self.product_data)} products with ratings")
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def filter_by_ratings(self, products_df):
        """Filter products based on database ratings"""
        if products_df.empty:
            return products_df
        
        # Calculate weighted rating score
        products_df['rating_score'] = (
            products_df['product_rating'] * 0.6 + 
            products_df['overall_rating'] * 0.4
        )
        
        # Filter products with rating score above average
        avg_rating = products_df['rating_score'].mean()
        filtered_products = products_df[
            products_df['rating_score'] >= avg_rating
        ].copy()
        
        if not filtered_products.empty:
            filtered_products = filtered_products.sort_values(
                'rating_score', 
                ascending=False
            )
        
        return filtered_products

    def get_recommendations(self, user_id, city):
        """Generate recommendations using database ratings"""
        # Get weather information
        weather_info = self.weather_analyzer.get_detailed_weather(city)
        if not weather_info:
            return None
            
        # Get user activity data
        activity_data = self.user_tracker.get_user_activity(user_id)
        
        # Get weather-based products
        weather_products, weather_category = self.get_weather_based_products(weather_info)
        
        # Apply rating-based filtering to weather products
        weather_products = self.filter_by_ratings(weather_products)
        
        # Personalize recommendations based on user activity
        if activity_data:
            interested_categories = activity_data['category_interests']
            if interested_categories:
                # Prioritize products from user's interested categories
                category_weights = {cat: count/sum(interested_categories.values()) 
                                 for cat, count in interested_categories.items()}
                
                # Get products from interested categories
                interest_based_products = self.product_data[
                    self.product_data['product_category_tree'].isin(interested_categories.keys())
                ]
                
                # Apply rating filtering
                interest_based_products = self.filter_by_ratings(interest_based_products)
                
                # Combine and deduplicate recommendations
                recommendations = pd.concat([
                    weather_products, 
                    interest_based_products
                ]).drop_duplicates()
                
                # Calculate final score combining relevance and ratings
                recommendations['relevance_score'] = recommendations['product_category_tree'].map(
                    lambda x: category_weights.get(x, 0)
                )
                recommendations['final_score'] = (
                    recommendations['relevance_score'] * 0.4 + 
                    recommendations['rating_score'] * 0.6
                )
                recommendations = recommendations.sort_values('final_score', ascending=False)
            else:
                recommendations = weather_products
                
            # Limit recommendations
            recommendations = recommendations.head(10)
        else:
            recommendations = weather_products.sample(n=min(5, len(weather_products)))
            
        # Add rating analysis summary
        analysis_summary = {
            'average_product_rating': recommendations['product_rating'].mean(),
            'average_overall_rating': recommendations['overall_rating'].mean(),
            'average_rating_score': recommendations['rating_score'].mean(),
            'highest_rated_product': {
                'name': recommendations.loc[recommendations['rating_score'].idxmax(), 'product_name'],
                'rating_score': recommendations['rating_score'].max()
            }
        }
       
        return {
            'city': city,
            'recommendations': recommendations,
            'weather': weather_info,  # Now includes coordinates
            'activity': activity_data,
            'weather_category': weather_category,
            'rating_analysis': analysis_summary
        }

    def analyze_user_sentiment(self, user_id, user_text=None, user_image=None, user_voice=None):
        """Analyze user sentiment across available modalities"""
        return self.sentiment_analyzer.analyze_multimodal(
            user_id, text=user_text, image_path=user_image, audio_path=user_voice
        )
    
    def get_contextual_factors(self, city):
        """Get contextual factors for recommendations using coordinates from weather data"""
        # Get weather data first to obtain coordinates
        weather_info = self.weather_analyzer.get_detailed_weather(city)
        
        if not weather_info or 'coordinates' not in weather_info:
            # If we can't get coordinates, fall back to default behavior
            return self.context_engine.get_context_factors(city)
        
        # Extract latitude from weather data
        latitude = weather_info['coordinates']['latitude']
        
        # Use the latitude to get context factors
        return self.context_engine.get_context_factors(city, latitude)
        
    def apply_context_to_recommendations(self, recommendations, context_factors):
        """Apply contextual factors to refine recommendations"""
        if recommendations.empty:
            return recommendations
            
        # Create a context score column
        recommendations['context_score'] = 0.0
        
        # Apply seasonal trends
        season_trends = context_factors['season']['trends']
        if season_trends:
            for trend in season_trends:
                # Increase score for products matching seasonal trends
                mask = recommendations['product_category_tree'].str.contains(trend, case=False, na=False)
                recommendations.loc[mask, 'context_score'] += 0.2
        
        # Apply regional preferences
        regional_prefs = context_factors['regional_preferences']
        if regional_prefs:
            for pref in regional_prefs:
                # Increase score for products matching regional preferences
                mask = recommendations['product_category_tree'].str.contains(pref, case=False, na=False)
                recommendations.loc[mask, 'context_score'] += 0.15
        
        # Apply upcoming event trends
        for event in context_factors['upcoming_events']:
            # Weight based on how soon the event is
            event_weight = 0.3 * (1 - min(event['days_away'] / 7, 1))
            
            for trend in event['trends']:
                # Increase score for products matching event trends
                mask = recommendations['product_category_tree'].str.contains(trend, case=False, na=False)
                recommendations.loc[mask, 'context_score'] += event_weight
        
        # Combine with existing scores or create final score
        if 'final_score' in recommendations.columns:
            # If final score exists, include context score with appropriate weight
            recommendations['final_score'] = (
                recommendations['final_score'] * 0.7 + 
                recommendations['context_score'] * 0.3
            )
        else:
            # Otherwise create a new score combining rating and context
            if 'rating_score' in recommendations.columns:
                recommendations['final_score'] = (
                    recommendations['rating_score'] * 0.6 + 
                    recommendations['context_score'] * 0.4
                )
            else:
                recommendations['final_score'] = recommendations['context_score']
        
        # Sort by the updated final score
        recommendations = recommendations.sort_values('final_score', ascending=False)
        
        return recommendations
    
    def apply_sentiment_to_recommendations(self, recommendations, user_id):
        """Refine recommendations based on user sentiment"""
        # Get sentiment trends
        sentiment_trends = self.sentiment_analyzer.get_sentiment_trends(user_id)
        
        if not sentiment_trends or recommendations.empty:
            return recommendations
            
        # Adjust recommendations based on sentiment
        avg_sentiment = sentiment_trends['average_sentiment']
        
        # Define product categories that match different sentiment states
        positive_categories = ['entertainment', 'luxury', 'premium', 'gift']
        negative_categories = ['comfort', 'self-care', 'relaxation', 'wellness']
        
        # Create a sentiment adjustment score
        recommendations['sentiment_adj'] = 0.0
        
        if avg_sentiment > 0.3:  # Positive sentiment
            # Boost products in positive categories
            for category in positive_categories:
                mask = recommendations['product_category_tree'].str.contains(
                    category, case=False, na=False
                )
                recommendations.loc[mask, 'sentiment_adj'] += 0.2 * avg_sentiment
        elif avg_sentiment < -0.1:  # Negative sentiment
            # Boost products in comfort/wellness categories
            for category in negative_categories:
                mask = recommendations['product_category_tree'].str.contains(
                    category, case=False, na=False
                )
                recommendations.loc[mask, 'sentiment_adj'] += 0.2 * abs(avg_sentiment)
        
        # Apply sentiment adjustment to final score
        if 'final_score' in recommendations.columns:
            recommendations['final_score'] += recommendations['sentiment_adj']
        else:
            recommendations['final_score'] = recommendations['sentiment_adj']
            
        # Sort by updated score
        recommendations = recommendations.sort_values('final_score', ascending=False)
        
        return recommendations
    
    def get_enhanced_recommendations(self, user_id, city, user_text=None, user_image=None, user_voice=None):
        """Generate recommendations with all enhanced features using weather API coordinates"""
        # Get basic recommendations first
        basic_results = self.get_recommendations(user_id, city)
        
        if not basic_results:
            return None
            
        recommendations = basic_results['recommendations']

        if user_image and os.path.exists(user_image):
            image_labels = self.sentiment_analyzer.extract_labels_from_image(user_image)
            recommendations = recommendations[
                recommendations['product_name'].str.contains('|'.join(image_labels), case=False, na=False) |
                recommendations['product_category_tree'].str.contains('|'.join(image_labels), case=False, na=False)
            ]

        
        # Extract latitude from weather data if available
        latitude = None
        if 'weather' in basic_results and 'coordinates' in basic_results['weather']:
            latitude = basic_results['weather']['coordinates']['latitude']
        
        # Get contextual factors using the city and extracted latitude
        context_factors = self.get_contextual_factors(city)
        
        # Apply context awareness
        recommendations = self.apply_context_to_recommendations(recommendations, context_factors)
        
        # Analyze sentiment if modalities provided
        if user_text or user_image or user_voice:
            sentiment_results = self.analyze_user_sentiment(
                user_id, user_text, user_image, user_voice
            )
            # Apply sentiment analysis
            recommendations = self.apply_sentiment_to_recommendations(recommendations, user_id)
        else:
            sentiment_results = None
        
        # Update the results dictionary
        enhanced_results = basic_results.copy()
        enhanced_results['recommendations'] = recommendations
        enhanced_results['context_factors'] = context_factors
        enhanced_results['sentiment_analysis'] = sentiment_results
        
        return enhanced_results
def test_enhanced_features():
    # Initialize the system
    recommender = EnhancedRecommendationSystem(WEATHER_API_KEY)
    
    # Load product data
    recommender.load_data('flipkart_com-ecommerce_sample.csv')
    
    # Enable multimodal sentiment analysis (with mock implementations)
    recommender.sentiment_analyzer.enable_image_analysis("mock_key")
    recommender.sentiment_analyzer.enable_voice_analysis("mock_key")
    
    # Simulate user
    user_id = 'test_enhanced_user'
    
    # Track various user activities (as in your existing code)
    recommender.user_tracker.track_activity(user_id, 'product_view', 
        {'product_id': 'P1', 'category': 'winter wear', 'time_spent': 45})
    
    recommender.user_tracker.track_activity(user_id, 'browsing', 
        {'url': '/winter-collection', 'time_spent': 120, 'scroll_depth': 0.8})
    
    recommender.user_tracker.track_activity(user_id, 'search', 
        {'query': 'waterproof boots'})
    
    # Simulate user text for sentiment analysis
    user_text = "I'm really excited about finding some great new products for the rainy season!"
    
    # Get enhanced recommendations
    results = recommender.get_enhanced_recommendations(
        user_id, 'Kolkata',
        user_text=user_text, user_image="mock_image.jpg", user_voice="mock_audio.mp3"
    )
    
    if results:
        print("\n=== Enhanced Real-time Recommendation Results ===")
        
        print(f"\nWeather Conditions in {results['city']}:")
        print(f"Temperature: {results['weather']['temperature']}°C")
        print(f"Condition: {results['weather']['condition']}")
        
        print(f"\nContextual Factors:")
        print(f"Season: {results['context_factors']['season']['season']}")
        print(f"Seasonal Trends: {results['context_factors']['season']['trends']}")
        print(f"Region: {results['context_factors']['region']}")
        print(f"Regional Preferences: {results['context_factors']['regional_preferences']}")
        
        if results['context_factors']['upcoming_events']:
            print(f"Upcoming Events:")
            for event in results['context_factors']['upcoming_events']:
                print(f"  - {event['name']} (in {event['days_away']} days)")
        
        if results['sentiment_analysis']:
            print(f"\nMultimodal Sentiment Analysis:")
            print(f"Modalities Used: {results['sentiment_analysis']['modalities_used']}")
            if 'text_sentiment' in results['sentiment_analysis']:
                print(f"Text Sentiment: {results['sentiment_analysis']['text_sentiment']['compound']:.2f}")
            if 'image_sentiment' in results['sentiment_analysis']:
                print(f"Image Sentiment: Joy={results['sentiment_analysis']['image_sentiment']['joy']:.2f}")
            if 'voice_sentiment' in results['sentiment_analysis']:
                print(f"Voice Tone: {results['sentiment_analysis']['voice_sentiment']['tone']}")
            if 'combined_sentiment' in results['sentiment_analysis']:
                print(f"Combined Sentiment: {results['sentiment_analysis']['combined_sentiment']:.2f}")
        
        print(f"\nTop 5 Recommended Products (Enhanced):")
        top_recommendations = results['recommendations'].head(10)
        for i, (_, product) in enumerate(top_recommendations.iterrows(), 1):
            print(f"{i}. {product['product_name']} - Score: {product.get('final_score', 0):.2f}")
def update_context_data():
    """Update contextual awareness data in real-time"""
    logger.info("Updating contextual data...")
    
    try:
        # Update events calendar with any new events
        recommender.context_engine._initialize_events_calendar()
        
        # You could fetch seasonal trends from an API here
        # Example: fetch_seasonal_trends_api()
        
        # Update regional preferences based on sales data
        # This could connect to your sales database
        # Example: update_regional_preferences_from_sales()
        
        logger.info("Contextual data updated successfully")
    except Exception as e:
        logger.error(f"Error updating contextual data: {e}")

def process_sentiment_signals():
    """Process new sentiment signals from users in real-time"""
    logger.info("Processing sentiment signals...")
    
    try:
        # In a real system, this would:
        # 1. Check for new user comments/reviews
        # 2. Process new images uploaded by users
        # 3. Analyze any voice interactions
        
        # For demonstration, let's simulate processing for active users
        active_users = list(recommender.user_tracker.user_sessions.keys())
        
        for user_id in active_users:
            # Simulate text from recent user activity
            recent_searches = recommender.user_tracker.user_sessions[user_id].get('search_history', [])
            if recent_searches:
                # Use the most recent search as text for sentiment analysis
                sample_text = recent_searches[-1]
                
                # Process the sentiment
                recommender.sentiment_analyzer.analyze_multimodal(
                    user_id, 
                    text=sample_text,
                    # Image and voice would come from user uploads in a real system
                )
                
        logger.info(f"Processed sentiment signals for {len(active_users)} active users")
    except Exception as e:
        logger.error(f"Error processing sentiment signals: {e}")


def monitor_recommendation_system():
    """Monitor the performance and health of the recommendation system"""
    logger.info("Monitoring system performance...")
    
    # Track system resources
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    
    # Log performance metrics
    logger.info(f"System CPU: {cpu_usage}%, Memory: {memory_usage}%")
    
    # Track recommendation metrics
    total_users = len(recommender.user_tracker.user_sessions)
    active_sessions = sum(1 for uid, session in recommender.user_tracker.user_sessions.items() 
                        if (datetime.now() - session['last_activity']).total_seconds() < 3600)
    
    logger.info(f"Active users: {active_sessions}/{total_users}")
    
    # Alert if system resources are critically high
    if cpu_usage > 90 or memory_usage > 90:
        logger.warning("ALERT: System resources critically high!")
        # In a real system, this could send an alert to administrators
        # Example: send_alert_email("System resources critical")

def enhanced_schedule_updates(interval_minutes=5):
    """Schedule all real-time updates including context and sentiment processing"""
    # Schedule the original recommendation updates
    schedule.every(interval_minutes).minutes.do(run_recommendation_system)
    
    # Schedule context data updates (daily)
    schedule.every(24).hours.do(update_context_data)
    
    # Schedule sentiment signal processing (more frequent)
    schedule.every(interval_minutes).minutes.do(process_sentiment_signals)
    
    print(f"Starting enhanced real-time updates:")
    print(f"- Recommendations every {interval_minutes} minutes")
    print(f"- Context data updates daily")
    print(f"- Sentiment processing every {interval_minutes} minutes")
    
    while True:
        schedule.run_pending()
        time.sleep(1)
# # Initialize the system
# recommender = EnhancedRecommendationSystem(WEATHER_API_KEY)

# # Load your product dataset
# recommender.load_data('flipkart_com-ecommerce_sample.csv')

# # Simulate various user activities
# user_id = 'test_user'

# # Simulate browsing behavior
# recommender.user_tracker.track_activity(user_id, 'browsing', {
#     'url': '/summer-collection',
#     'time_spent': 180,
#     'scroll_depth': 0.75,
#     'clicks': 12,
#     'mouse_movement': 1500
# })

# # Simulate product views
# recommender.user_tracker.track_activity(user_id, 'product_view', {
#     'product_id': 'P123',
#     'category': 'Summer wear',
#     'time_spent': 45
# })

# # Simulate search
# recommender.user_tracker.track_activity(user_id, 'search', {
#     'query': 'waterproof boots'
# })

# # activity_tracker = EnhancedUserActivityTracker('website_classification.csv')
# # activity_tracker.track_activity()

# # Get recommendations
# results = recommender.get_recommendations(user_id, 'Kolkata')

# print("\n\n")
# test_enhanced_features()

# # Display results
# if results:
#     print("\n=== Recommendation Results ===")
#     print(f"\nWeather Conditions:")
#     print(json.dumps(results['city'], indent=2))
#     print(json.dumps(results['weather'], indent=2))
#     print(f"\nUser Activity:")
#     print(json.dumps(results['activity'], indent=2))
#     print(f"\nRating Analysis:")
#     print(json.dumps(results['rating_analysis'], indent=2))
#     print(f"\nRecommended Products:")
#     print(results['recommendations'][['product_name', 'product_category_tree']])


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.width', None)

# # Display the full DataFrame for recommended products
# display(results['recommendations'])
# Initialize the system
recommender = EnhancedRecommendationSystem(WEATHER_API_KEY)

# Load your product dataset
recommender.load_data('flipkart_com-ecommerce_sample.csv')

# Initialize context engine and sentiment analyzer
recommender.context_engine.load_regional_preferences()
recommender.sentiment_analyzer.enable_image_analysis("mock_key") 
recommender.sentiment_analyzer.enable_voice_analysis("mock_key")  

# Simulate various user activities
user_id = 'test_user'

# Simulate browsing behavior
recommender.user_tracker.track_activity(user_id, 'browsing', {
    'url': '/summer-collection',
    'time_spent': 180,
    'scroll_depth': 0.75,
    'clicks': 12,
    'mouse_movement': 1500
})

# Simulate product views
recommender.user_tracker.track_activity(user_id, 'product_view', {
    'product_id': 'P123',
    'category': 'Summer wear',
    'time_spent': 45
})

# Simulate search
recommender.user_tracker.track_activity(user_id, 'search', {
    'query': 'waterproof boots'
})

# Get recommendations
results = recommender.get_recommendations(user_id, 'Kolkata')

# Get enhanced recommendations with context and sentiment
enhanced_results = recommender.get_enhanced_recommendations(
    user_id, 
    'Kolkata',
    user_text="I'm looking for something to keep me dry during monsoon season",
    user_image="mock_image.jpg", user_voice="mock_audio.mp3"
)

# Display results
if results:
    print("\n=== Basic Recommendation Results ===")
    print(f"\nWeather Conditions:")
    print(json.dumps(results['city'], indent=2))
    print(json.dumps(results['weather'], indent=2))
    print(f"\nUser Activity:")
    print(json.dumps(results['activity'], indent=2))
    print(f"\nRating Analysis:")
    print(json.dumps(results['rating_analysis'], indent=2))
    print(f"\nRecommended Products:")
    print(results['recommendations'][['product_name', 'product_category_tree']])

if enhanced_results:
    print("\n=== Enhanced Recommendation Results with Context & Sentiment ===")
    print(f"\nContextual Factors:")
    print(f"Season: {enhanced_results['context_factors']['season']['season']}")
    print(f"Region: {enhanced_results['context_factors']['region']}")
    
    if enhanced_results['sentiment_analysis']:
        print(f"\nSentiment Analysis:")
        print(f"Modalities Used: {enhanced_results['sentiment_analysis']['modalities_used']}")
        if 'combined_sentiment' in enhanced_results['sentiment_analysis']:
            print(f"Combined Sentiment: {enhanced_results['sentiment_analysis']['combined_sentiment']:.2f}")
    
    print(f"\nTop Enhanced Recommendations:")
    print(enhanced_results['recommendations'].head(5)[['product_name', 'product_category_tree', 'final_score']])

# Schedule the real-time updates (uncomment to run continuously)
# enhanced_schedule_updates(interval_minutes=5)

# For demo purposes, run one-time updates instead of continuous scheduling
update_context_data()
process_sentiment_signals()
monitor_recommendation_system()

print("\nReal-time update functions executed successfully.")
