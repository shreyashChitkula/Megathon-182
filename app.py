from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import config
import mysql.connector as connector
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import bcrypt
from collections import Counter
from dotenv import load_dotenv
import json
from datetime import datetime
import cv2

# Import our enhanced analysis modules
from fraud_detection import FraudDetector
from multimodal_analysis import MultimodalAnalyzer
from claim_confidence import ClaimConfidenceCalculator
from explainability import ExplainabilityGenerator
from text_attention import TextAttentionAnalyzer

# Import advanced deep learning modules
from gradcam_explainer import GradCAMExplainer, SimpleGradCAM
from advanced_fraud import HybridFraudDetector
from vehicle_consistency_checker import VehicleConsistencyChecker

# Import rule-based severity (PRIMARY - interpretable & explainable)
from rule_based_severity import RuleBasedSeverityAssessor, convert_yolo_to_detections, SeverityLevel

import torch


load_dotenv()


app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production-' + os.urandom(24).hex())

# Initialize analysis modules
fraud_detector = FraudDetector()
multimodal_analyzer = MultimodalAnalyzer()
confidence_calculator = ClaimConfidenceCalculator()
explainability_gen = ExplainabilityGenerator()
text_attention_analyzer = TextAttentionAnalyzer()

# Initialize advanced deep learning modules
print("Initializing advanced AI modules...")
advanced_fraud_detector = HybridFraudDetector(use_forensics=True)
vehicle_consistency_checker = VehicleConsistencyChecker()

# Initialize rule-based severity assessor (PRIMARY - interpretable & explainable)
rule_based_severity = RuleBasedSeverityAssessor()
print("✅ Advanced modules initialized (rule-based severity active)")


def connect_to_db():
    try:
        connection = connector.connect(**config.mysql_credentials)
        return connection
    except connector.Error as e:
        print(f"Error connecting to database: {e}")
        return None


def calculate_cost_band(total_cost, cost_multiplier=1.0):
    """Calculate cost band and risk level based on total cost"""
    estimated_cost = total_cost * cost_multiplier
    
    if estimated_cost < 20000:
        band = 'LOW'
        risk = 'Low'
    elif estimated_cost < 50000:
        band = 'MEDIUM'
        risk = 'Medium'
    elif estimated_cost < 100000:
        band = 'HIGH'
        risk = 'High'
    else:
        band = 'CRITICAL'
        risk = 'Very High'
    
    return {
        'band': band,
        'risk': risk,
        'base_cost': total_cost,
        'estimated_cost': estimated_cost,
        'cost_multiplier': cost_multiplier
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        password = request.form.get('password')
        email = request.form.get('email')
        vehicle_id = request.form.get('vehicleId')
        contact_number = request.form.get('phoneNumber')
        address = request.form.get('address')
        car_brand = request.form.get('carBrand')
        model = request.form.get('carModel')
        
        # print("DATA from form")
        # print(f"name : {name}")
        # print(f"email : {email}")
        # print(f"password : {password}")
        # print(f"vehicle_id : {vehicle_id}")
        # print(f"contact_number : {contact_number}")
        # print(f"address : {address}")
        # print(f"car_brand : {car_brand}")
        # print(f"model : {model}")

        if not all([name, password, email, vehicle_id, contact_number, address, car_brand, model]):
            flash("All fields are required!", "error")
            return render_template('signup.html')

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        connection = connect_to_db()
        if connection:
            try:
                with connection.cursor() as cursor:
                    query = '''
                    INSERT INTO user_info (name, password, email, vehicle_id, contact_number, address, car_brand, model)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    '''
                    cursor.execute(query, (name, hashed_password, email, vehicle_id, contact_number, address, car_brand, model))
                    connection.commit()
                flash("Signup successful!", "success")
                return redirect(url_for('dashboard'))
            except connector.IntegrityError as e:
                if 'Duplicate entry' in str(e):
                    flash("Email already exists. Please use a different email.", "error")
                else:
                    flash("An error occurred while signing up. Please try again.", "error")
            except connector.Error as e:
                print(f"Error executing query: {e}")
                flash("An error occurred while signing up. Please try again.", "error")
        else:
            flash("Database connection failed. Please try again later.", "error")
            
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        print(f"Email : {email}")
        print(f"Password : {password}")

        if not email or not password:
            flash("Email and password are required!", "error")
            return render_template('login.html')

        connection = connect_to_db()
        if connection:
            try:
                with connection.cursor() as cursor:
                    query = "SELECT password FROM user_info WHERE email = %s"
                    cursor.execute(query, (email,))
                    result = cursor.fetchone()
                    if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
                        session['user_email'] = email  # Store user email in session
                        flash("Login successful!", "success")
                        return redirect(url_for('dashboard'))
                    else:
                        flash("Invalid email or password.", "error")
            except connector.Error as e:
                print(f"Error executing query: {e}")
                flash("An error occurred during login. Please try again.", "error")
        else:
            flash("Database connection failed. Please try again later.", "error")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_email', None)  # Remove user email from session
    flash("You have been logged out.", "info")
    
    return redirect(url_for('login'))


# Load YOLO model with device configuration (FINE-TUNED MODEL)
model_path = os.path.join(os.path.dirname(__file__), "models", "fine-tuned.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading fine-tuned YOLO model on device: {device}")
model = YOLO(model_path)
print(f"✓ Model loaded with {len(model.names)} classes: {list(model.names.values())}")

# Enable GPU acceleration if available
if torch.cuda.is_available():
    print("GPU acceleration enabled")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("Running on CPU")


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        # Receive uploaded images (multiple)
        files = request.files.getlist('images')
        claim_description = request.form.get('description', '')
        
        if not files or len(files) == 0:
            flash('No files uploaded!', 'error')
            return render_template('dashboard.html')
        
        # Save all uploaded images
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_dir, exist_ok=True)
        
        uploaded_images = []
        for idx, file in enumerate(files):
            if file and file.filename:
                image_path = os.path.join(static_dir, f'uploaded_image_{idx}.jpg')
                file.save(image_path)
                uploaded_images.append(image_path)
                print(f"Image {idx + 1} uploaded successfully: {file.filename}")
        
        if len(uploaded_images) == 0:
            flash('No valid images uploaded!', 'error')
            return render_template('dashboard.html')
        
        # ============ VEHICLE CONSISTENCY CHECK ============
        # Check if all uploaded images belong to the same vehicle
        vehicle_consistency = None
        if len(uploaded_images) > 1:
            vehicle_consistency = vehicle_consistency_checker.check_consistency(uploaded_images)
            
            # Print detailed report
            report = vehicle_consistency_checker.generate_consistency_report(vehicle_consistency)
            print(report)
            
            # Warning if inconsistent (but don't block - could be different angles)
            if not vehicle_consistency['is_consistent']:
                print("\n⚠️ WARNING: Images may not be from the same vehicle!")
                print(f"   Confidence: {vehicle_consistency['confidence']:.1%}")
                print(f"   Reason: {vehicle_consistency['reason']}\n")
        else:
            vehicle_consistency = {
                'is_consistent': True,
                'confidence': 1.0,
                'reason': 'Single image uploaded',
                'pairwise_scores': [],
                'warnings': []
            }
        
        # Process all images and aggregate detections
        all_class_counts = Counter()
        all_detected_images = []
        per_image_detections = []  # Store detections per image
        
        for idx, img_path in enumerate(uploaded_images):
            # Make predictions using YOLO with aggressive NMS to remove duplicates
            result = model(
                img_path,
                conf=0.25,      # Confidence threshold
                iou=0.4,        # IoU threshold for NMS (lower = more aggressive)
                max_det=50,     # Maximum detections
                agnostic_nms=False  # Class-specific NMS (removes same-class overlaps)
            )
            detected_objects = result[0].boxes
            class_ids = [box.cls.item() for box in detected_objects]
            
            # Store per-image detection data
            per_image_detections.append({
                'index': idx,
                'class_ids': class_ids,
                'num_detections': len(class_ids),
                'boxes': detected_objects
            })
            
            # Aggregate counts from this image
            all_class_counts.update(class_ids)
            
            # Save detected image with thinner boxes
            detected_image_path = os.path.join(static_dir, f'detected_image_{idx}.jpg')
            plotted_img = result[0].plot(
                line_width=1,        # Thinner bounding box lines
                font_size=10,        # Smaller font for labels
                labels=True,
                conf=True,
                boxes=True
            )
            
            cv2.imwrite(detected_image_path, plotted_img)
            all_detected_images.append(f'detected_image_{idx}.jpg')
            print(f"Processed image {idx + 1}: {len(class_ids)} detections")
        
        # Use aggregated counts for analysis
        class_counts = all_class_counts
        
        # Handle no damage detected case (but continue to generate report)
        no_damage_detected = len(class_counts) == 0
        if no_damage_detected:
            print("⚠️ No damage detected - generating clean vehicle report")
        else:
            print(f"Total images processed: {len(uploaded_images)}")
            print(f"Aggregated detections: {dict(class_counts)}")
        
        # ============ MULTI-IMAGE ANALYSIS ============
        # Analyze each image individually and track worst case
        print("\n" + "="*60)
        print("ANALYZING ALL IMAGES INDIVIDUALLY")
        print("="*60)
        
        all_image_analyses = []
        worst_severity_score = 0
        worst_severity_image_idx = 0
        highest_fraud_risk = 0
        highest_fraud_image_idx = 0
        
        for idx, img_path in enumerate(uploaded_images):
            print(f"\n--- Analyzing Image {idx + 1} ---")
            
            # Get detection data for this specific image
            img_detections = per_image_detections[idx]
            num_detections = img_detections['num_detections']
            
            # Individual image analysis result
            img_analysis = {
                'index': idx,
                'path': img_path,
                'num_detections': num_detections
            }
            
            # 1. Severity analysis for this image
            try:
                # Simple rule-based severity score
                # More detections = higher severity
                img_severity_score = min(num_detections * 15, 100)
                img_analysis['severity_score'] = img_severity_score
                
                if img_severity_score > worst_severity_score:
                    worst_severity_score = img_severity_score
                    worst_severity_image_idx = idx
                
                print(f"  Detections: {num_detections}")
                print(f"  Severity Score: {img_severity_score}/100")
            except Exception as e:
                print(f"  Severity analysis failed: {e}")
                img_analysis['severity_score'] = 0
            
            # 2. Basic fraud check for this image
            try:
                # Simple fraud score based on consistency
                # Low detections when others are high could be suspicious
                fraud_score = 0
                
                # If this image has significantly fewer detections than others, flag it
                if len(uploaded_images) > 1:
                    avg_detections = sum(d['num_detections'] for d in per_image_detections) / len(per_image_detections)
                    if num_detections < avg_detections * 0.3:  # Less than 30% of average
                        fraud_score = 20  # Mild suspicion
                
                img_analysis['fraud_checked'] = True
                img_analysis['fraud_score'] = fraud_score
                
                if fraud_score > highest_fraud_risk:
                    highest_fraud_risk = fraud_score
                    highest_fraud_image_idx = idx
                    
                print(f"  Fraud Risk: {fraud_score}/100")
            except Exception as e:
                print(f"  Fraud check failed: {e}")
                img_analysis['fraud_checked'] = False
            
            all_image_analyses.append(img_analysis)
            print(f"  ✓ Image {idx + 1} analysis complete")
        
        # Summary of all analyses
        print(f"\n{'='*60}")
        print(f"MULTI-IMAGE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total images analyzed: {len(all_image_analyses)}")
        for idx, analysis in enumerate(all_image_analyses):
            print(f"  Image {idx + 1}:")
            print(f"    - Detections: {analysis['num_detections']}")
            print(f"    - Severity: {analysis['severity_score']}/100")
            print(f"    - Fraud: {analysis['fraud_score']}/100")
        print(f"{'='*60}\n")
        
        # Use worst-case image as primary for detailed analysis
        image_path = uploaded_images[worst_severity_image_idx]
        print(f"✓ Using Image {worst_severity_image_idx + 1} as primary (worst severity: {worst_severity_score}/100)")
        print("="*60 + "\n")
        
        # Get the user's email from session
        user_email = session.get('user_email')
        print(user_email)
        if not user_email:
            flash('You need to log in to get an estimate.', 'error')
            return redirect(url_for('login'))

        # Fetch part prices from the database
        part_prices = get_part_prices(user_email, class_counts)
        
        # ============ ENHANCED MULTIMODAL ANALYSIS ============
        
        # 1. Fraud Detection
        fraud_analysis = {
            'duplicate': fraud_detector.check_duplicate_images(image_path, user_email),
            'ai_generated': fraud_detector.detect_ai_generated(image_path),
            'claim_frequency': fraud_detector.analyze_claim_frequency(user_email)
        }
        
        # 2. Text-Image Consistency (if description provided)
        if claim_description:
            consistency_analysis = multimodal_analyzer.check_text_image_consistency(
                claim_description, part_prices
            )
            authenticity_analysis = multimodal_analyzer.analyze_claim_authenticity(claim_description)
            text_severity = multimodal_analyzer.extract_severity_indicators(claim_description)
            
            # NEW: Text Attention Analysis - highlight important words
            detected_part_names = [get_part_name_from_id(class_id) for class_id in class_counts.keys()]
            detected_part_names = [name for name in detected_part_names if name]  # Filter None values
            text_attention = text_attention_analyzer.analyze_text_attention(
                claim_description, detected_part_names
            )
        else:
            consistency_analysis = {'consistency_score': 0.5, 'verdict': 'No description provided'}
            authenticity_analysis = {'authenticity_score': 0.5, 'issues': ['No description provided']}
            text_severity = {'severity': 'unknown', 'confidence': 0}
            text_attention = {
                'highlighted_html': '<p style="color: #718096;">No description provided</p>',
                'statistics': {},
                'insights': []
            }
        
        # 3. Damage Severity Analysis (ENHANCED with CNN)
        # Handle no damage detected case
        if no_damage_detected:
            # Create clean vehicle severity report
            severity_analysis_enhanced = {
                'severity': 'NONE',
                'severity_label': 'No Damage',
                'severity_score': 0,
                'cost_multiplier': 1.0,
                'explanation': 'No damage detected - vehicle appears clean',
                'has_critical_damage': False,
                'critical_parts_damaged': []
            }
            severity_analysis = {
                'severity': 'NONE',
                'severity_label': 'No Damage',
                'severity_score': 0,
                'cost_multiplier': 1.0,
                'explanation': 'No visible damage found in the uploaded images',
                'has_critical_damage': False,
                'critical_parts_damaged': []
            }
            severity_to_use = severity_analysis_enhanced
            print("✓ Severity: NONE (No damage detected)")
        else:
            # ===== NEW: Rule-Based Severity Assessment (Interpretable!) =====
            # Convert YOLO detections to rule-based format
            try:
                # Get image dimensions from first result
                img = cv2.imread(image_path)
                img_height, img_width = img.shape[:2]
                
                # Convert detections
                detections_for_rules = convert_yolo_to_detections(
                    result[0], img_width, img_height
                )
                
                # Assess severity with rule-based system
                rule_based_result = rule_based_severity.assess(detections_for_rules)
                
                print(f"✓ Rule-Based Severity: {rule_based_result.severity_level.name} "
                      f"(Score: {rule_based_result.severity_score:.3f}, "
                      f"Confidence: {rule_based_result.confidence:.2%})")
                print(f"  Explanation: {rule_based_result.explanation}")
                
                # Create enhanced result with rule-based data
                severity_analysis_rule_based = {
                    'severity': rule_based_result.severity_level.name,
                    'severity_label': rule_based_result.severity_level.name.title(),
                    'severity_score': int(rule_based_result.severity_score * 100),
                    'cost_multiplier': 1.0 + (rule_based_result.severity_score * 0.5),  # 1.0 to 1.5x
                    'explanation': rule_based_result.explanation,
                    'rule_based_result': {
                        'level': rule_based_result.severity_level.name,
                        'score': float(rule_based_result.severity_score),
                        'confidence': float(rule_based_result.confidence),
                        'damage_count': rule_based_result.damage_count,
                        'total_damage_area': float(rule_based_result.total_damage_area * 100),  # Convert to %
                        'critical_damage': rule_based_result.critical_damage,
                        'factors': {k: float(v) for k, v in rule_based_result.factors.items()},
                        'breakdown': rule_based_result.damage_breakdown
                    },
                    'has_critical_damage': rule_based_result.critical_damage,
                    'critical_parts_damaged': [
                        d['damage_type'] for d in rule_based_result.damage_breakdown 
                        if d['is_critical']
                    ]
                }
                
                # Use rule-based severity (primary and only method)
                severity_to_use = severity_analysis_rule_based
                severity_analysis_enhanced = None  # No CNN
                print("✅ Using rule-based severity (interpretable & explainable)")
                
            except Exception as e:
                print(f"❌ Rule-based severity error: {e}")
                # Set default severity on error
                severity_analysis_rule_based = {
                    'level': 'MODERATE',
                    'score': 50,
                    'explanation': f'Error in severity assessment: {str(e)}',
                    'confidence': 0
                }
                severity_to_use = severity_analysis_rule_based
                severity_analysis_enhanced = None
        
        # 4. Cost Band Estimation with edge case handling
        try:
            if part_prices:
                total_cost = sum(part.get('total', 0) for part in part_prices.values())
            else:
                total_cost = 0
            
            if no_damage_detected:
                # Clean vehicle - no repair costs
                cost_band = {
                    'band': 'NO DAMAGE',
                    'risk': 'None',
                    'base_cost': 0,
                    'estimated_cost': 0,
                    'cost_multiplier': 1.0
                }
                print("✓ Cost: ₹0 (No damage detected)")
            else:
                cost_multiplier = severity_to_use.get('cost_multiplier') or severity_analysis.get('cost_multiplier', 1.0)
                cost_band = calculate_cost_band(total_cost, cost_multiplier)
        except Exception as e:
            print(f"⚠ Cost band calculation error: {e}")
            cost_band = {
                'band': None,
                'risk': 'Unknown',
                'base_cost': total_cost if 'total_cost' in locals() else 0,
                'estimated_cost': total_cost if 'total_cost' in locals() else 0,
                'cost_multiplier': 1.0
            }
        
        # 4.5. Advanced Forensics Analysis for ALL images
        all_forensics_results = []
        all_ela_images = []
        
        if advanced_fraud_detector.forensics_detector:
            print(f"\n🔬 Running forensics analysis for all {len(uploaded_images)} image(s)...")
            
            for idx, img_path in enumerate(uploaded_images):
                print(f"  Analyzing Image {idx + 1}...")
                forensics_result = advanced_fraud_detector.forensics_detector.detect_manipulation(img_path)
                
                # Store ELA image path if it exists
                if forensics_result.get('ela_visualization_path'):
                    ela_path = forensics_result['ela_visualization_path']
                    # Extract just the filename
                    ela_filename = ela_path.replace('static/', '') if 'static/' in ela_path else ela_path
                    all_ela_images.append(ela_filename)
                else:
                    all_ela_images.append(None)
                
                all_forensics_results.append(forensics_result)
                print(f"  ✓ Forensics for Image {idx + 1}: {'MANIPULATED' if forensics_result['is_manipulated'] else 'AUTHENTIC'} "
                      f"({forensics_result['confidence']:.2%})")
            
            # Use worst image forensics for main fraud analysis
            fraud_analysis['advanced_forensics'] = all_forensics_results[worst_severity_image_idx]
        else:
            print("⚠ Advanced forensics unavailable")
            fraud_analysis['advanced_forensics'] = None
            all_ela_images = [None] * len(uploaded_images)
        
        # 5. Calculate Overall Confidence Score
        analysis_results = {
            'detected_parts': part_prices,
            'consistency': consistency_analysis,
            'authenticity': authenticity_analysis,
            'fraud': fraud_analysis,
            'severity': severity_to_use,  # Use enhanced severity
            'severity_enhanced': severity_analysis_enhanced,  # Keep both for comparison
            'cost_band': cost_band
        }
        
        if no_damage_detected:
            # High confidence for clean vehicle (no damage)
            # Match the structure from ClaimConfidenceCalculator
            confidence_result = {
                'confidence_score': 95,
                'recommendation': 'APPROVE - No damage detected',
                'verdict': 'HIGH CONFIDENCE - Vehicle appears clean',
                'risk_level': 'LOW',
                'component_scores': {
                    # Core scores from ClaimConfidenceCalculator
                    'text_image_consistency': 100,      # Perfect match
                    'authenticity': 100,                # Authentic image
                    'fraud_indicators': 100,            # No fraud detected
                    'damage_plausibility': 100,         # Clean vehicle is plausible
                    'claim_frequency': 100,             # Normal claim pattern
                    # Additional scores that template may use
                    'detection_confidence': 95,         # High detection confidence
                    'fraud_risk': 0,                    # Zero fraud risk
                    'consistency': 100,                 # Overall consistency
                    'severity_appropriateness': 100,    # Severity correctly assessed
                    'image_quality': 95,                # Good image quality
                    'claim_consistency': 100,           # Consistent claim
                    'forensics_score': 100              # No manipulation
                },
                'explanations': [
                    'No damage detected in uploaded images',
                    'Vehicle appears to be in clean condition',
                    'No repair costs required',
                    '✓ High confidence in clean vehicle status'
                ]
            }
            print("✓ Confidence: 95% (Clean vehicle)")
        else:
            confidence_result = confidence_calculator.calculate_confidence_score(analysis_results)
        
        # 6. Generate Explainability Visualizations
        print("Generating explainability visualizations...")
        
        # Note: CNN-based Grad-CAM removed - using YOLO-based attention instead
        all_gradcam_images = [None] * len(uploaded_images)
        
        # Generate visualizations for ALL images
        all_explanation_images = []
        all_heatmap_images = []
        
        print(f"\n🎨 Generating explainability visualizations for all {len(uploaded_images)} image(s)...")
        
        for idx, img_path in enumerate(uploaded_images):
            print(f"  Processing visualizations for Image {idx + 1}...")
            
            # Get YOLO result for this image (re-run detection for explainability)
            result_for_viz = model(img_path, conf=0.25, iou=0.4)
            
            # Generate explanation and heatmap
            explanation_img = explainability_gen.generate_explanation_image(
                img_path, result_for_viz, part_prices
            )
            heatmap_img = explainability_gen.generate_heatmap(img_path, result_for_viz)
            
            # Save with unique names
            explanation_path = os.path.join(static_dir, f'explanation_annotated_{idx}.jpg')
            heatmap_path = os.path.join(static_dir, f'heatmap_{idx}.jpg')
            
            cv2.imwrite(explanation_path, cv2.cvtColor(explanation_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
            
            all_explanation_images.append(f'explanation_annotated_{idx}.jpg')
            all_heatmap_images.append(f'heatmap_{idx}.jpg')
            
            print(f"  ✓ Visualizations for Image {idx + 1} complete")
        
        # Generate textual explanation (based on overall analysis)
        if no_damage_detected:
            textual_explanation = (
                "✅ CLEAN VEHICLE REPORT\n\n"
                "No damage detected in the uploaded image(s).\n"
                "The AI system has analyzed all images and found no visible damage, "
                "dents, scratches, or broken parts.\n\n"
                "Analysis Summary:\n"
                f"• Images Analyzed: {len(uploaded_images)}\n"
                "• Damage Detected: None\n"
                "• Estimated Repair Cost: ₹0\n"
                "• Vehicle Condition: Clean\n\n"
                "This is a positive result indicating the vehicle appears to be in good condition."
            )
        else:
            textual_explanation = explainability_gen.generate_textual_explanation(
                {**analysis_results, 'confidence': confidence_result}
            )
        
        # Save report (using worst severity image)
        report_path = os.path.join(static_dir, 'analysis_report.png')
        primary_explanation = explainability_gen.generate_explanation_image(
            uploaded_images[worst_severity_image_idx], 
            model(uploaded_images[worst_severity_image_idx], conf=0.25, iou=0.4), 
            part_prices
        )
        primary_heatmap = explainability_gen.generate_heatmap(
            uploaded_images[worst_severity_image_idx], 
            model(uploaded_images[worst_severity_image_idx], conf=0.25, iou=0.4)
        )
        explainability_gen.save_explanation_report(report_path, analysis_results, primary_explanation, primary_heatmap)
        
        # 7. Store claim in database
        store_claim_analysis(user_email, image_path, claim_description, analysis_results, confidence_result)
        
        # Print comprehensive analysis
        print("\n" + "="*60)
        print("COMPREHENSIVE CLAIM ANALYSIS")
        print("="*60)
        print(textual_explanation)
        print("="*60 + "\n")
        
        # Prepare image lists for template
        uploaded_image_names = [f'uploaded_image_{i}.jpg' for i in range(len(uploaded_images))]
        
        # Debug logging
        print(f"\n📸 TEMPLATE DATA:")
        print(f"  Total images: {len(uploaded_images)}")
        print(f"  Uploaded names: {uploaded_image_names}")
        print(f"  Detected images: {all_detected_images}")
        print(f"  Analysis data: {len(all_image_analyses)} records")
        print(f"  Worst severity index: {worst_severity_image_idx}\n")
        
        return render_template('estimate.html', 
                             original_image=f'uploaded_image_{worst_severity_image_idx}.jpg',  # Worst severity image
                             uploaded_images=uploaded_image_names,   # All uploaded images
                             detected_image=f'detected_image_{worst_severity_image_idx}.jpg',  # Worst severity detected
                             detected_images=all_detected_images,    # All detected images
                             all_image_analyses=all_image_analyses,  # Per-image analysis results
                             worst_severity_idx=worst_severity_image_idx,  # Which image is worst
                             explanation_image=f'explanation_annotated_{worst_severity_image_idx}.jpg',  # Primary explanation
                             heatmap_image=f'heatmap_{worst_severity_image_idx}.jpg',  # Primary heatmap
                             explanation_images=all_explanation_images,  # All explanation images
                             heatmap_images=all_heatmap_images,  # All heatmap images
                             gradcam_image=all_gradcam_images[worst_severity_image_idx] if all_gradcam_images and all_gradcam_images[worst_severity_image_idx] else None,  # Primary grad-cam
                             gradcam_images=all_gradcam_images,  # All Grad-CAM images
                             ela_images=all_ela_images,  # All ELA forensics images
                             analysis_report='analysis_report.png',
                             part_prices=part_prices,
                             severity=severity_to_use,  # FIXED: Use rule-based result (most interpretable)
                             severity_enhanced=severity_analysis_enhanced,
                             cost_band=cost_band,
                             confidence=confidence_result,
                             fraud_analysis=fraud_analysis,
                             consistency=consistency_analysis,
                             text_attention=text_attention,
                             claim_description=claim_description,
                             textual_explanation=textual_explanation,
                             num_images=len(uploaded_images),
                             vehicle_consistency=vehicle_consistency,
                             no_damage_detected=no_damage_detected)  # Flag for clean vehicle

    return render_template('dashboard.html')


def get_part_prices(email, class_counts):
    """Get part prices with proper error handling and defaults"""
    connection = connect_to_db()
    if not connection:
        print("⚠ Database connection failed")
        return {}
    
    try:
        with connection.cursor(dictionary=True) as cursor:
            # Get user's car brand and model
            cursor.execute("SELECT car_brand, model FROM user_info WHERE email = %s", (email,))
            user_data = cursor.fetchone()
            
            if not user_data:
                print(f"⚠ User not found: {email}")
                return {}
            
            car_brand = user_data.get('car_brand')
            car_model = user_data.get('model')
            
            if not car_brand or not car_model:
                print("⚠ User profile incomplete (missing car brand/model)")
                return {}
            
            # Fetch part prices
            prices = {}
            for class_id, count in class_counts.items():
                detailed_part_name = get_part_name_from_id(class_id)
                
                if not detailed_part_name:
                    print(f"⚠ Unknown part class ID: {class_id}")
                    continue
                
                # Map detailed name to generic category for pricing
                generic_part_name = map_detailed_to_generic_part(detailed_part_name)
                
                try:
                    cursor.execute(
                        "SELECT price FROM car_models WHERE brand = %s AND model = %s AND part = %s",
                        (car_brand, car_model, generic_part_name)
                    )
                    price_data = cursor.fetchone()
                    
                    if price_data and price_data.get('price') is not None:
                        price_per_part = float(price_data['price'])
                        total_price = price_per_part * count
                        # Use detailed name as key for display, but price from generic category
                        prices[detailed_part_name] = {
                            'count': int(count),
                            'price': price_per_part,
                            'total': total_price,
                            'generic_category': generic_part_name
                        }
                    else:
                        print(f"⚠ No price data for {car_brand} {car_model} - {generic_part_name} (mapped from {detailed_part_name})")
                        # Use default fallback price (₹10,000 per part)
                        default_price = 10000.0
                        prices[detailed_part_name] = {
                            'count': int(count),
                            'price': default_price,
                            'total': default_price * count,
                            'generic_category': generic_part_name
                        }
                except Exception as part_error:
                    print(f"⚠ Error fetching price for {detailed_part_name}: {part_error}")
                    continue
            
            if not prices:
                print("⚠ No price data found for any detected parts")
            else:
                print(f"✓ Retrieved prices for {len(prices)} part(s)")
            
            return prices
            
    except connector.Error as e:
        print(f"⚠ Database error: {e}")
        return {}
    except Exception as e:
        print(f"⚠ Unexpected error in get_part_prices: {e}")
        return {}
    finally:
        if connection:
            connection.close()


def get_part_name_from_id(class_id):
    """Map class ID to part name for fine-tuned model (17 classes)"""
    class_names = {
        0: 'Bodypanel-Dent',
        1: 'Front-Windscreen-Damage',
        2: 'Headlight-Damage',
        3: 'Rear-windscreen-Damage',
        4: 'RunningBoard-Dent',
        5: 'Sidemirror-Damage',
        6: 'Signlight-Damage',
        7: 'Taillight-Damage',
        8: 'bonnet-dent',
        9: 'boot-dent',
        10: 'doorouter-dent',
        11: 'fender-dent',
        12: 'front-bumper-dent',
        13: 'pillar-dent',
        14: 'quaterpanel-dent',
        15: 'rear-bumper-dent',
        16: 'roof-dent'
    }
    return class_names.get(int(class_id), None)


def map_detailed_to_generic_part(detailed_part_name):
    """
    Map detailed part names from fine-tuned model to generic categories for pricing.
    The database has 7 generic categories: Bonnet, Bumper, Dickey, Door, Fender, Light, Windshield
    """
    mapping = {
        'Bodypanel-Dent': 'Fender',
        'Front-Windscreen-Damage': 'Windshield',
        'Headlight-Damage': 'Light',
        'Rear-windscreen-Damage': 'Windshield',
        'RunningBoard-Dent': 'Fender',
        'Sidemirror-Damage': 'Light',  # Mirrors often priced similar to lights
        'Signlight-Damage': 'Light',
        'Taillight-Damage': 'Light',
        'bonnet-dent': 'Bonnet',
        'boot-dent': 'Dickey',
        'doorouter-dent': 'Door',
        'fender-dent': 'Fender',
        'front-bumper-dent': 'Bumper',
        'pillar-dent': 'Fender',
        'quaterpanel-dent': 'Fender',
        'rear-bumper-dent': 'Bumper',
        'roof-dent': 'Fender'
    }
    return mapping.get(detailed_part_name, 'Fender')  # Default to Fender if unknown


def store_claim_analysis(user_email, image_path, description, analysis_results, confidence_result):
    """Store claim analysis in database for fraud tracking and history"""
    connection = connect_to_db()
    if connection:
        try:
            with connection.cursor() as cursor:
                # Compute image hash for duplicate detection
                image_hash = fraud_detector.compute_image_hash(image_path)
                
                # Insert into claim_history
                cursor.execute("""
                    INSERT INTO claim_history 
                    (email, image_hash, claim_description, confidence_score, 
                     estimated_cost, severity_level, cost_band, fraud_flags, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_email,
                    image_hash,
                    description,
                    confidence_result['confidence_score'],
                    analysis_results['cost_band'].get('estimated_cost', 0),
                    analysis_results['severity'].get('severity', 'NONE'),
                    analysis_results['cost_band'].get('band') or analysis_results['cost_band'].get('cost_band', 'NO DAMAGE'),
                    json.dumps({
                        'duplicate': analysis_results['fraud']['duplicate'].get('is_duplicate', False),
                        'ai_generated': analysis_results['fraud']['ai_generated'].get('is_ai_generated', False),
                        'suspicious_frequency': analysis_results['fraud']['claim_frequency'].get('is_suspicious', False)
                    }),
                    confidence_result.get('recommendation', 'APPROVE').split(' - ')[0]  # APPROVE/REVIEW/REJECT/INVESTIGATE
                ))
                
                claim_id = cursor.lastrowid
                
                # Insert detailed analysis
                cursor.execute("""
                    INSERT INTO claim_analysis
                    (claim_id, detected_parts, text_image_consistency, authenticity_score,
                     duplicate_detected, ai_generated_detected, damage_severity,
                     fraud_indicators, confidence_breakdown, explanation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    claim_id,
                    json.dumps(analysis_results.get('detected_parts', {}), default=str),
                    analysis_results.get('consistency', {}).get('consistency_score', 0.5),
                    analysis_results.get('authenticity', {}).get('authenticity_score', 0.5),
                    analysis_results.get('fraud', {}).get('duplicate', {}).get('is_duplicate', False),
                    analysis_results.get('fraud', {}).get('ai_generated', {}).get('is_ai_generated', False),
                    json.dumps(analysis_results.get('severity', {}), default=str),
                    json.dumps(analysis_results.get('fraud', {}), default=str),
                    json.dumps(confidence_result.get('component_scores', {}), default=str),
                    '\n'.join(confidence_result.get('explanations', ['No explanation available']))
                ))
                
                connection.commit()
                print(f"Claim {claim_id} stored successfully")
                
        except connector.Error as e:
            print(f"Error storing claim analysis: {e}")
        finally:
            connection.close()


@app.route('/view_profile')
def view_profile():
    if 'user_email' not in session:
        flash('You need to login to view your profile.', 'error')
        return redirect(url_for('login'))

    connection = connect_to_db()
    if connection:
        try:
            with connection.cursor(dictionary=True) as cursor:
                # Fetch current user information
                cursor.execute("SELECT * FROM user_info WHERE email = %s", (session['user_email'],))
                user_info = cursor.fetchone()
                if not user_info:
                    flash('User not found.', 'error')
                    return redirect(url_for('dashboard'))
                return render_template('view_profile.html', user_info=user_info)
        except connector.Error as e:
            print(f"Error executing query: {e}")
            flash("An error occurred while fetching your profile. Please try again.", "error")
    else:
        flash("Database connection failed. Please try again later.", "error")

    return redirect(url_for('dashboard'))


@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_email' not in session:
        flash('You need to login to edit your profile.', 'error')
        return redirect(url_for('login'))

    connection = connect_to_db()
    if connection:
        try:
            with connection.cursor(dictionary=True) as cursor:
                if request.method == 'POST':
                    # Update user information
                    query = '''
                    UPDATE user_info
                    SET name = %s, email = %s, vehicle_id = %s, contact_number = %s, 
                        address = %s, car_brand = %s, model = %s
                    WHERE email = %s
                    '''
                    cursor.execute(query, (
                        request.form['name'],
                        request.form['email'],
                        request.form['vehicleId'],
                        request.form['phoneNumber'],
                        request.form['address'],
                        request.form['carBrand'],
                        request.form['carModel'],
                        session['user_email']
                    ))
                    connection.commit()
                    flash('Profile updated successfully!', 'success')
                    session['user_email'] = request.form['email']  # Update session if email changed
                    return redirect(url_for('dashboard'))

                # Fetch current user information
                cursor.execute("SELECT * FROM user_info WHERE email = %s", (session['user_email'],))
                user_info = cursor.fetchone()
                return render_template('edit_profile.html', user_info=user_info)

        except connector.Error as e:
            print(f"Error executing query: {e}")
            flash("An error occurred while updating your profile. Please try again.", "error")
    else:
        flash("Database connection failed. Please try again later.", "error")

    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)