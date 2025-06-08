import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def find_data_file():
    """Find the correct path to the data file"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Possible paths relative to script location
    possible_paths = [
        os.path.join(script_dir, '..', 'data', 'data', 'bank-additional-full.csv'),  # From app/ folder
        os.path.join(script_dir, 'data', 'data', 'bank-additional-full.csv'),        # From root folder
        'data/data/bank-additional-full.csv',  # Current directory
        '../data/data/bank-additional-full.csv'  # One level up
    ]
    
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

def simple_term_deposit_prediction():
    """Simple version of term deposit prediction"""
    print("=== TERM DEPOSIT PREDICTION ===")
    print("Loading data...")
    
    # Find the data file
    data_path = find_data_file()
    if not data_path:
        return None
    
    try:
        # Load data
        df = pd.read_csv(data_path, sep=';')
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Quick EDA
        print(f"\n📊 Target distribution:")
        target_counts = df['y'].value_counts()
        print(target_counts)
        success_rate = target_counts['yes']/len(df)*100
        print(f"Success rate: {success_rate:.1f}%")
        
        # Simple preprocessing
        print("\n🔧 Preprocessing data...")
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop('y')
        
        print(f"Encoding {len(categorical_cols)} categorical variables...")
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
        
        # Prepare features and target
        X = df_processed.drop('y', axis=1)
        y = df_processed['y'].map({'yes': 1, 'no': 0})
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train model
        print("\n🤖 Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        print("\n📈 Model Performance:")
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Top 5 Most Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Simple insights
        print("\n💡 Key Insights:")
        avg_duration_yes = df[df['y'] == 'yes']['duration'].mean()
        avg_duration_no = df[df['y'] == 'no']['duration'].mean()
        print(f"  📞 Subscribers have longer call duration: {avg_duration_yes:.0f}s vs {avg_duration_no:.0f}s")
        
        avg_age_yes = df[df['y'] == 'yes']['age'].mean()
        avg_age_no = df[df['y'] == 'no']['age'].mean()
        print(f"  👥 Subscribers average age: {avg_age_yes:.1f} vs non-subscribers: {avg_age_no:.1f}")
        
        print("\n✅ Model training complete!")
        print("="*50)
        
        return model
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = simple_term_deposit_prediction()
    if model:
        print("🎉 SUCCESS: Model trained successfully!")
    else:
        print("💥 FAILED: Check the errors above")
