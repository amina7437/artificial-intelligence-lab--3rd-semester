from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ------------------------------------------------------------------
# 1. Load CSV – try current folder, then absolute path
# ------------------------------------------------------------------
possible_paths = [
    'student_burnout_dataset.csv',
    'student_burnout_dataset (1).csv',
    'C:/Users/SYS/Desktop/Student Burnout And Performance Measure/student_burnout_dataset (1).csv'
]
df = None
for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ Loading file: {path}")
        df = pd.read_csv(path)
        break

if df is None:
    print("❌ ERROR: CSV file not found.")
    print("Please place the CSV file in the same folder as app.py")
    print("Expected names:", possible_paths[:2])
    exit(1)

print(f"✅ Loaded {len(df)} records")

# ------------------------------------------------------------------
# 2. Data cleaning
# ------------------------------------------------------------------
required_cols = ['student_id', 'age', 'gender', 'study_hours_per_day', 'sleep_hours',
                 'social_media_hours', 'attendance_percentage', 'assignment_completion_rate',
                 'stress_level', 'burnout_level', 'exam_score']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    exit(1)

df = df.dropna().reset_index(drop=True)
df = df[df['exam_score'] >= 0]
print(f"✅ After cleaning: {len(df)} records")

# ------------------------------------------------------------------
# 3. Encode categorical variables
# ------------------------------------------------------------------
feature_columns = ['study_hours_per_day', 'sleep_hours', 'social_media_hours',
                   'attendance_percentage', 'assignment_completion_rate', 'stress_level']

label_encoders = {}
for col in ['gender', 'burnout_level']:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# ------------------------------------------------------------------
# 4. Prepare features & targets
# ------------------------------------------------------------------
X = df[feature_columns + ['gender_encoded']].copy()
y_reg = df['exam_score']
y_clf = df['burnout_level_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------
# 5. Train models
# ------------------------------------------------------------------
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_r, y_train_r)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_c, y_train_c)

reg_score = reg_model.score(X_test_r, y_test_r)
clf_acc = accuracy_score(y_test_c, clf_model.predict(X_test_c))
print(f"✅ Regression R²: {reg_score:.3f}")
print(f"✅ Classification accuracy: {clf_acc:.3f}")

feature_names = feature_columns + ['gender']
feature_importance = dict(zip(feature_names, reg_model.feature_importances_))

# ------------------------------------------------------------------
# 6. Routes
# ------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard-data')
def dashboard_data():
    stats = {
        'total_students': int(len(df)),
        'avg_exam_score': float(round(df['exam_score'].mean(), 2)),
        'avg_study_hours': float(round(df['study_hours_per_day'].mean(), 2)),
        'avg_sleep_hours': float(round(df['sleep_hours'].mean(), 2)),
        'burnout_distribution': df['burnout_level'].value_counts().to_dict(),
        'avg_by_burnout': df.groupby('burnout_level')['exam_score'].mean().round(2).to_dict(),
        'correlations': {
            'study_hours_correlation': float(round(df['study_hours_per_day'].corr(df['exam_score']), 3)),
            'sleep_correlation': float(round(df['sleep_hours'].corr(df['exam_score']), 3)),
            'social_media_correlation': float(round(df['social_media_hours'].corr(df['exam_score']), 3)),
            'attendance_correlation': float(round(df['attendance_percentage'].corr(df['exam_score']), 3)),
            'assignment_correlation': float(round(df['assignment_completion_rate'].corr(df['exam_score']), 3))
        },
        'model_performance': {
            'r2_score': float(round(reg_score, 3)),
            'accuracy': float(round(clf_acc, 3))
        }
    }
    return jsonify(stats)

@app.route('/api/student-analysis')
def student_analysis():
    student_id = request.args.get('student_id', type=int)
    if student_id is not None and student_id in df['student_id'].values:
        row = df[df['student_id'] == student_id].iloc[0]
        return jsonify({
            'student_id': int(row['student_id']),
            'age': int(row['age']),
            'gender': str(row['gender']),
            'study_hours': float(row['study_hours_per_day']),
            'sleep_hours': float(row['sleep_hours']),
            'social_media_hours': float(row['social_media_hours']),
            'attendance': float(row['attendance_percentage']),
            'assignment_completion': float(row['assignment_completion_rate']),
            'stress_level': int(row['stress_level']),
            'burnout_level': str(row['burnout_level']),
            'exam_score': float(row['exam_score'])
        })
    else:
        students_list = []
        for _, row in df.head(100).iterrows():
            students_list.append({
                'student_id': int(row['student_id']),
                'age': int(row['age']),
                'gender': str(row['gender']),
                'exam_score': float(row['exam_score']),
                'burnout_level': str(row['burnout_level'])
            })
        return jsonify({'students': students_list, 'total': len(df)})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        study_hours = float(data.get('study_hours', 6))
        sleep_hours = float(data.get('sleep_hours', 7))
        social_media = float(data.get('social_media_hours', 2))
        attendance = float(data.get('attendance', 75))
        assignment_completion = float(data.get('assignment_completion', 70))
        stress_level = int(data.get('stress_level', 5))
        gender_str = data.get('gender', 'Male')

        gender_encoded = label_encoders['gender'].transform([gender_str])[0]

        features = np.array([[
            study_hours, sleep_hours, social_media,
            attendance, assignment_completion, stress_level,
            gender_encoded
        ]])
        features_scaled = scaler.transform(features)

        pred_score = reg_model.predict(features_scaled)[0]
        pred_burnout_encoded = clf_model.predict(features_scaled)[0]
        burnout_level = label_encoders['burnout_level'].inverse_transform([pred_burnout_encoded])[0]

        risk_factors = []
        if sleep_hours < 6: risk_factors.append("Low sleep hours (<6) – Risk of burnout")
        if stress_level > 7: risk_factors.append("High stress level (>7) – Critical risk")
        if study_hours > 8 and sleep_hours < 7: risk_factors.append("High study hours with low sleep – Burnout alert")
        if social_media > 4: risk_factors.append("High social media usage (>4hrs) – May affect performance")
        if attendance < 70: risk_factors.append("Low attendance (<70%) – Performance risk")
        if assignment_completion < 50: risk_factors.append("Low assignment completion (<50%) – Academic risk")

        recommendations = []
        if pred_score < 40:
            recommendations.append("⚠️ Critical: Consider academic counseling immediately")
        elif pred_score < 55:
            recommendations.append("📚 Need improvement: Increase study efficiency and consistency")
        else:
            recommendations.append("🎉 Excellent! Keep maintaining your good habits")

        if sleep_hours < 6:
            recommendations.append("😴 Improve sleep to 7–8 hours for better performance and health")
        if pred_burnout_encoded == 2:
            recommendations.append("🚨 High burnout risk detected! Take regular breaks and seek support")
        elif pred_burnout_encoded == 1:
            recommendations.append("⚠️ Moderate burnout signs – Practice self-care and time management")
        if stress_level > 7:
            recommendations.append("🧘 Consider stress management techniques like meditation or exercise")
        if not recommendations:
            recommendations.append("✅ Good balance! Maintain current habits for continued success")

        return jsonify({
            'success': True,
            'predicted_exam_score': round(float(pred_score), 1),
            'predicted_burnout_level': burnout_level,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/insights')
def insights():
    try:
        high = df[df['exam_score'] > 70]
        low = df[df['exam_score'] < 35]
        optimal_study = float(round(high['study_hours_per_day'].mean(), 1)) if len(high) > 0 else 6.0
        optimal_sleep = float(round(high['sleep_hours'].mean(), 1)) if len(high) > 0 else 7.0
        low_study = float(round(low['study_hours_per_day'].mean(), 1)) if len(low) > 0 else 4.0
        burnout_risk = df[(df['stress_level'] > 8) & (df['sleep_hours'] < 6)]
        burnout_alert = float(round(len(burnout_risk) / len(df) * 100, 1))
        return jsonify({
            'optimal_study_hours': optimal_study,
            'optimal_sleep_hours': optimal_sleep,
            'low_performer_study_hours': low_study,
            'burnout_alert': burnout_alert,
            'high_stress_count': int(len(df[df['stress_level'] > 7])),
            'feature_importance': {k: float(v) for k, v in feature_importance.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ------------------------------------------------------------------
# 7. Run the app (no automatic browser opening)
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎓 STUDENT BURNOUT & PERFORMANCE SYSTEM")
    print("="*50)
    print("✅ Server is starting...")
    print("🌐 Open your browser and go to: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)