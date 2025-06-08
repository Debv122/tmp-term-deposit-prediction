import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

# Find the data file : This function will search for the data file in the specified locations.
def find_data_file():
    """Find the correct path to the data file"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Possible paths relative to script location
    possible_paths = [
        os.path.join(script_dir, '..', 'data', 'data', 'bank-additional-full.csv'),  # From app/ folder
        os.path.join(script_dir, 'data', 'data', 'bank-additional-full.csv'),        # From root folder
        'data/data/bank-additional-full.csv',  
        '../data/data/bank-additional-full.csv' 
    ]
    
    #  Loop through the possible paths and return the first one that exists
    for path in possible_paths:
        full_path = os.path.abspath(path)
        if os.path.exists(full_path):
            print(f"✅ Found data file at: {full_path}")
            return full_path
    
    print("❌ Data file not found!")
    print(f"Script location: {script_dir}")
    print("Searched paths:")
    for path in possible_paths:
        print(f"  - {os.path.abspath(path)}")
    return None

# Define the class for term deposit prediction
class TermDepositPredictor:
    def __init__(self, data_path=None):
        if data_path is None:
            self.data_path = find_data_file()
            if self.data_path is None:
                raise FileNotFoundError("Could not find the data file. Please check if 'bank-additional-full.csv' exists in the data directory.")
        else:
            self.data_path = data_path
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    # Load data
    def load_data(self):
        """Load the dataset"""
        print("Loading data...")
        try:
            self.df = pd.read_csv(self.data_path, sep=';')
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    # Preprocess data
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print("\n1. Dataset Overview:")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Data types and missing values
        print("\n2. Data Types and Missing Values:")
        info_df = pd.DataFrame({
            'Data Type': self.df.dtypes,
            'Missing Values': self.df.isnull().sum(),
            'Missing %': (self.df.isnull().sum() / len(self.df)) * 100
        })
        print(info_df)
        
        # Check for outliers in numerical columns
        print("\n3. Outlier Detection:")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)]
            print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.1f}%)")
        
        # Target variable distribution
        print("\n4. Target Variable Distribution:")
        target_counts = self.df['y'].value_counts()
        print(target_counts)
        print(f"Class distribution: {target_counts.values[1]/len(self.df)*100:.1f}% yes, {target_counts.values[0]/len(self.df)*100:.1f}% no")
        
        # Numerical features summary
        print("\n5. Numerical Features Summary:")
        print(self.df[numerical_cols].describe())
        
        # Categorical features
        print("\n6. Categorical Features:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'y':
                print(f"\n{col}: {self.df[col].nunique()} unique values")
                print(self.df[col].value_counts().head())
        
        # Create visualizations
        self.create_eda_plots()
        
    def create_eda_plots(self):
        """Create EDA visualizations"""
        # Main EDA plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Target distribution
        self.df['y'].value_counts().plot(kind='bar', ax=axes[0,0], title='Target Distribution')
        axes[0,0].set_xlabel('Subscribed')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # Age distribution by target
        self.df.boxplot(column='age', by='y', ax=axes[0,1])
        axes[0,1].set_title('Age Distribution by Target')
        
        # Duration vs Target
        self.df.boxplot(column='duration', by='y', ax=axes[1,0])
        axes[1,0].set_title('Call Duration by Target')
        
        # Campaign vs Target
        self.df.boxplot(column='campaign', by='y', ax=axes[1,1])
        axes[1,1].set_title('Number of Campaigns by Target')
        
        plt.tight_layout()
        plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation heatmap for numerical features
        plt.figure(figsize=(12, 10))
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional categorical analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Job vs Target
        job_target = pd.crosstab(self.df['job'], self.df['y'], normalize='index')
        job_target.plot(kind='bar', ax=axes[0,0], title='Subscription Rate by Job')
        axes[0,0].set_xlabel('Job')
        axes[0,0].set_ylabel('Proportion')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Education vs Target
        edu_target = pd.crosstab(self.df['education'], self.df['y'], normalize='index')
        edu_target.plot(kind='bar', ax=axes[0,1], title='Subscription Rate by Education')
        axes[0,1].set_xlabel('Education')
        axes[0,1].set_ylabel('Proportion')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Marital vs Target
        marital_target = pd.crosstab(self.df['marital'], self.df['y'], normalize='index')
        marital_target.plot(kind='bar', ax=axes[1,0], title='Subscription Rate by Marital Status')
        axes[1,0].set_xlabel('Marital Status')
        axes[1,0].set_ylabel('Proportion')
        axes[1,0].tick_params(axis='x', rotation=0)
        
        # Contact vs Target
        contact_target = pd.crosstab(self.df['contact'], self.df['y'], normalize='index')
        contact_target.plot(kind='bar', ax=axes[1,1], title='Subscription Rate by Contact Type')
        axes[1,1].set_xlabel('Contact Type')
        axes[1,1].set_ylabel('Proportion')
        axes[1,1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Handle categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop('y')  # Exclude target
        
        print(f"Encoding categorical variables: {list(categorical_cols)}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        
        # Separate features and target
        X = df_processed.drop('y', axis=1)
        y = df_processed['y'].map({'yes': 1, 'no': 0})
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train the predictive model"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Try both Random Forest and Logistic Regression
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        }
        
        best_model = None
        best_score = 0
        model_results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            if name == 'Logistic Regression':
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='f1')
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            model_results[name] = {
                'model': model,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            
            print(f"{name} Results:")
            print(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print(f"Test F1-Score: {f1:.4f}")
            print(f"Test AUC-ROC: {auc:.4f}")
            
            # Select best model based on F1-score (good for imbalanced data)
            if f1 > best_score:
                best_score = f1
                best_model = model
                self.model = model
                self.best_model_name = name
        
        self.model_results = model_results
        print(f"\nBest model: {self.best_model_name} with F1-Score: {best_score:.4f}")
        return self.model
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        if self.best_model_name == 'Logistic Regression':
            y_pred = self.model.predict(self.X_test_scaled)
            y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        else:
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Print detailed classification report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Create evaluation plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(self.y_test, y_pred_proba):.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Model comparison
        if hasattr(self, 'model_results'):
            print("\n" + "="*30)
            print("MODEL COMPARISON SUMMARY")
            print("="*30)
            
            comparison_df = pd.DataFrame({
                name: {
                    'CV F1 Mean': results['cv_f1_mean'],
                    'CV F1 Std': results['cv_f1_std'],
                    'Test Accuracy': results['accuracy'],
                    'Test Precision': results['precision'],
                    'Test Recall': results['recall'],
                    'Test F1': results['f1'],
                    'Test AUC': results['auc']
                }
                for name, results in self.model_results.items()
            }).T
            
            print(comparison_df.round(4))
        
        return y_pred, y_pred_proba
    
    def feature_importance(self):
        """Analyze feature importance"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        if hasattr(self.model, 'feature_importances_'):
            # For Random Forest
            feature_names = self.X_train.columns
            importances = self.model.feature_importances_
            
            # Create feature importance dataframe
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            print(feature_imp_df.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_imp_df.head(15), x='importance', y='feature')
            plt.title('Top 15 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_imp_df
        elif hasattr(self.model, 'coef_'):
            # For Logistic Regression
            feature_names = self.X_train.columns
            coefficients = self.model.coef_[0]
            
            # Create feature importance dataframe based on absolute coefficients
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
            print("Top 10 Most Important Features (by coefficient magnitude):")
            print(feature_imp_df.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_imp_df.head(15)
            sns.barplot(data=top_features, x='coefficient', y='feature')
            plt.title('Top 15 Feature Coefficients (Logistic Regression)')
            plt.xlabel('Coefficient Value')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_imp_df
        else:
            print("Feature importance not available for this model type.")
            return None
    
    def generate_insights(self):
        """Generate business insights"""
        print("\n" + "="*50)
        print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
        print("="*50)
        
        # Analyze successful vs unsuccessful campaigns
        success_rate = self.df['y'].value_counts(normalize=True)['yes']
        print(f"Overall Success Rate: {success_rate:.1%}")
        
        # Key insights from EDA
        print("\nKey Findings:")
        
        # Age analysis
        avg_age_yes = self.df[self.df['y'] == 'yes']['age'].mean()
        avg_age_no = self.df[self.df['y'] == 'no']['age'].mean()
        print(f"1. Average age of subscribers: {avg_age_yes:.1f} vs non-subscribers: {avg_age_no:.1f}")
        
        # Duration analysis
        avg_duration_yes = self.df[self.df['y'] == 'yes']['duration'].mean()
        avg_duration_no = self.df[self.df['y'] == 'no']['duration'].mean()
        print(f"2. Average call duration for subscribers: {avg_duration_yes:.0f}s vs non-subscribers: {avg_duration_no:.0f}s")
        
        # Campaign analysis
        avg_campaign_yes = self.df[self.df['y'] == 'yes']['campaign'].mean()
        avg_campaign_no = self.df[self.df['y'] == 'no']['campaign'].mean()
        print(f"3. Average number of contacts for subscribers: {avg_campaign_yes:.1f} vs non-subscribers: {avg_campaign_no:.1f}")
        
        # Job analysis
        job_success = self.df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean()).sort_values(ascending=False)
        print(f"\n4. Top 3 jobs with highest subscription rates:")
        for job, rate in job_success.head(3).items():
            print(f"   {job}: {rate:.1%}")
        
        # Education analysis
        edu_success = self.df.groupby('education')['y'].apply(lambda x: (x == 'yes').mean()).sort_values(ascending=False)
        print(f"\n5. Education levels with highest subscription rates:")
        for edu, rate in edu_success.items():
            print(f"   {edu}: {rate:.1%}")
        
        # Previous outcome analysis
        if 'poutcome' in self.df.columns:
            poutcome_success = self.df.groupby('poutcome')['y'].apply(lambda x: (x == 'yes').mean()).sort_values(ascending=False)
            print(f"\n6. Previous campaign outcome impact:")
            for outcome, rate in poutcome_success.items():
                print(f"   {outcome}: {rate:.1%}")
        
        print("\n" + "="*30)
        print("ACTIONABLE RECOMMENDATIONS")
        print("="*30)
        
        print("\n🎯 For Marketing Team:")
        print("1. 📞 Focus on call quality over quantity - longer conversations lead to better results")
        print("2. 🎨 Limit contact attempts - excessive calls reduce success rates")
        print("3. 👥 Target specific demographics based on age and job patterns")
        print("4. 📚 Consider education level in campaign personalization")
        print("5. 🔄 Leverage successful previous campaign outcomes")
        
        print("\n📊 For Campaign Optimization:")
        print("1. Develop scripts that encourage longer, meaningful conversations")
        print("2. Implement contact frequency limits to avoid customer fatigue")
        print("3. Create targeted campaigns for high-potential job categories")
        print("4. Use predictive model scores to prioritize leads")
        print("5. A/B test different approaches based on customer segments")
        
        # Model performance insights
        if hasattr(self, 'model_results'):
            best_f1 = self.model_results[self.best_model_name]['f1']
            best_precision = self.model_results[self.best_model_name]['precision']
            best_recall = self.model_results[self.best_model_name]['recall']
            
            print(f"\n🤖 Model Performance Insights:")
            print(f"• The {self.best_model_name} model achieves {best_f1:.1%} F1-score")
            print(f"• Precision: {best_precision:.1%} (of predicted subscribers, {best_precision:.1%} actually subscribe)")
            print(f"• Recall: {best_recall:.1%} (model identifies {best_recall:.1%} of actual subscribers)")
            
            if best_precision > best_recall:
                print("• Model is conservative - fewer false positives, may miss some opportunities")
            else:
                print("• Model is aggressive - catches more subscribers but with more false positives")
    
    def create_prediction_function(self):
        """Create a reusable prediction function"""
        def predict_new_client(client_data):
            """
            Predict subscription probability for a new client
            
            Parameters:
            client_data (dict): Dictionary with client features
            
            Returns:
            dict: Prediction results
            """
            try:
                # Convert to DataFrame
                client_df = pd.DataFrame([client_data])
                
                # Encode categorical variables
                for col in client_df.columns:
                    if col in self.label_encoders:
                        if client_data[col] in self.label_encoders[col].classes_:
                            client_df[col] = self.label_encoders[col].transform([client_data[col]])[0]
                        else:
                            # Handle unseen categories
                            client_df[col] = 0
                
                # Scale features if using Logistic Regression
                if self.best_model_name == 'Logistic Regression':
                    client_scaled = self.scaler.transform(client_df)
                    prediction = self.model.predict(client_scaled)[0]
                    probability = self.model.predict_proba(client_scaled)[0][1]
                else:
                    prediction = self.model.predict(client_df)[0]
                    probability = self.model.predict_proba(client_df)[0][1]
                
                return {
                    'prediction': 'Yes' if prediction == 1 else 'No',
                    'probability': probability,
                    'confidence': 'High' if probability > 0.7 or probability < 0.3 else 'Medium'
                }
            
            except Exception as e:
                return {'error': str(e)}
        
        return predict_new_client
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Term Deposit Prediction Analysis...")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # EDA
        self.exploratory_data_analysis()
        
        # Preprocessing
        self.preprocess_data()
        
        # Model training
        self.train_model()
        
        # Model evaluation
        self.evaluate_model()
        
        # Feature importance
        self.feature_importance()
        
        # Generate insights
        self.generate_insights()
        
        # Create prediction function
        self.predict_client = self.create_prediction_function()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📊 Best Model: {self.best_model_name}")
        print(f"🎯 F1-Score: {self.model_results[self.best_model_name]['f1']:.4f}")
        print(f"📈 AUC-ROC: {self.model_results[self.best_model_name]['auc']:.4f}")
        print("🔮 Prediction function ready for new clients!")
        
        return self.model

def main():
    """Main function to run the analysis"""
    try:
        # Initialize the predictor (it will automatically find the data file)
        predictor = TermDepositPredictor()
        
        # Run complete analysis
        model = predictor.run_complete_analysis()
        
        print("\n" + "="*50)
        print("🧪 TESTING PREDICTION FUNCTION")
        print("="*50)
        
        # Example prediction
        sample_client = {
            'age': 35,
            'job': 'management',
            'marital': 'married',
            'education': 'tertiary',
            'default': 'no',
            'housing': 'yes',
            'loan': 'no',
            'contact': 'cellular',
            'month': 'may',
            'day_of_week': 'mon',
            'duration': 300,
            'campaign': 2,
            'pdays': 999,
            'previous': 0,
            'poutcome': 'nonexistent',
            'emp.var.rate': 1.1,
            'cons.price.idx': 93.994,
            'cons.conf.idx': -36.4,
            'euribor3m': 4.857,
                       'nr.employed': 5191.0
        }
        
        result = predictor.predict_client(sample_client)
        
        print(f"Sample Client Prediction:")
        print(f"Client Profile: 35-year-old manager, married, tertiary education")
        print(f"Call duration: 300s, Campaign attempts: 2")
        print(f"Prediction: {result.get('prediction', 'Error')}")
        print(f"Probability: {result.get('probability', 0):.1%}")
        print(f"Confidence: {result.get('confidence', 'Unknown')}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        
        print("\n" + "="*50)
        print("📁 FILES GENERATED")
        print("="*50)
        print("✅ eda_plots.png - Exploratory data analysis visualizations")
        print("✅ correlation_heatmap.png - Feature correlation matrix")
        print("✅ categorical_analysis.png - Categorical feature analysis")
        print("✅ model_evaluation.png - Model performance visualizations")
        print("✅ feature_importance.png - Most important features chart")
        
        print("\n🎉 Analysis complete! Model is ready for deployment.")
        
        return predictor
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure the data file 'bank-additional-full.csv' exists in one of these locations:")
        print("- data/data/bank-additional-full.csv")
        print("- ../data/data/bank-additional-full.csv")
        print("\nYou can download the dataset from the UCI Machine Learning Repository:")
        print("https://archive.ics.uci.edu/ml/datasets/Bank+Marketing")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predictor = main()


