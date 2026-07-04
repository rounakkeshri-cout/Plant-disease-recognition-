import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Plant Disease Detection", layout="wide")

model = tf.keras.models.load_model("trained_model.h5")
# ---------------- LEAF DETECTOR (runs BEFORE disease model) ----------------
leaf_detector = tf.keras.models.load_model("leaf_detector.h5")

def is_leaf(uploaded_file, threshold=0.5):
    from PIL import Image
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    input_arr = np.array(image).astype("float32")
    input_arr = np.expand_dims(input_arr, axis=0)
    input_arr = tf.keras.applications.efficientnet_v2.preprocess_input(input_arr)
    not_leaf_probability = leaf_detector.predict(input_arr, verbose=0)[0][0]
    uploaded_file.seek(0)
    return not_leaf_probability < threshold
disease_info = {

# ---------------- APPLE ----------------
"Apple___Apple_scab": {
"type": "Fungal Disease",
"treatment": "Apply fungicides like Captan or Mancozeb",
"prevention": "Remove infected leaves and ensure airflow"
},
"Apple___Black_rot": {
"type": "Fungal Disease",
"treatment": "Prune infected parts and apply fungicide",
"prevention": "Avoid wounds and maintain hygiene"
},
"Apple___Cedar_apple_rust": {
"type": "Fungal Disease",
"treatment": "Use fungicides and remove nearby cedar trees",
"prevention": "Plant resistant varieties"
},
"Apple___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Maintain proper care"
},

# ---------------- BLUEBERRY ----------------
"Blueberry___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Maintain soil acidity and watering"
},

# ---------------- CHERRY ----------------
"Cherry_(including_sour)___Powdery_mildew": {
"type": "Fungal Disease",
"treatment": "Apply sulfur fungicide",
"prevention": "Improve air circulation"
},
"Cherry_(including_sour)___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Regular monitoring"
},

# ---------------- CORN ----------------
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
"type": "Fungal Disease",
"treatment": "Use fungicides like Azoxystrobin",
"prevention": "Crop rotation and residue management"
},
"Corn_(maize)___Common_rust_": {
"type": "Fungal Disease",
"treatment": "Apply fungicides",
"prevention": "Use resistant hybrids"
},
"Corn_(maize)___Northern_Leaf_Blight": {
"type": "Fungal Disease",
"treatment": "Use fungicide spray",
"prevention": "Rotate crops and remove debris"
},
"Corn_(maize)___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Maintain proper nutrients"
},

# ---------------- GRAPE ----------------
"Grape___Black_rot": {
"type": "Fungal Disease",
"treatment": "Apply fungicides like Myclobutanil",
"prevention": "Prune infected vines"
},
"Grape___Esca_(Black_Measles)": {
"type": "Fungal Disease",
"treatment": "Remove infected plants",
"prevention": "Avoid trunk injuries"
},
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
"type": "Fungal Disease",
"treatment": "Use fungicides",
"prevention": "Ensure airflow"
},
"Grape___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Maintain vineyard hygiene"
},

# ---------------- ORANGE ----------------
"Orange___Haunglongbing_(Citrus_greening)": {
"type": "Bacterial Disease",
"treatment": "No cure, remove infected plants",
"prevention": "Control psyllid insects"
},

# ---------------- PEACH ----------------
"Peach___Bacterial_spot": {
"type": "Bacterial Disease",
"treatment": "Use copper sprays",
"prevention": "Avoid leaf wetness"
},
"Peach___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Maintain orchard hygiene"
},

# ---------------- PEPPER ----------------
"Pepper,_bell___Bacterial_spot": {
"type": "Bacterial Disease",
"treatment": "Apply copper bactericides",
"prevention": "Use disease-free seeds"
},
"Pepper,_bell___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Maintain proper irrigation"
},

# ---------------- POTATO ----------------
"Potato___Early_blight": {
"type": "Fungal Disease",
"treatment": "Apply Mancozeb fungicide",
"prevention": "Crop rotation"
},
"Potato___Late_blight": {
"type": "Fungal Disease",
"treatment": "Use Metalaxyl fungicide",
"prevention": "Avoid wet conditions"
},
"Potato___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Healthy soil management"
},

# ---------------- RASPBERRY ----------------
"Raspberry___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Maintain proper care"
},

# ---------------- SOYBEAN ----------------
"Soybean___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Use quality seeds"
},

# ---------------- SQUASH ----------------
"Squash___Powdery_mildew": {
"type": "Fungal Disease",
"treatment": "Apply sulfur fungicide",
"prevention": "Avoid overcrowding"
},

# ---------------- STRAWBERRY ----------------
"Strawberry___Leaf_scorch": {
"type": "Fungal Disease",
"treatment": "Use fungicides",
"prevention": "Remove infected leaves"
},
"Strawberry___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Proper watering"
},

# ---------------- TOMATO ----------------
"Tomato___Bacterial_spot": {
"type": "Bacterial Disease",
"treatment": "Use copper bactericides",
"prevention": "Avoid overhead watering"
},
"Tomato___Early_blight": {
"type": "Fungal Disease",
"treatment": "Apply fungicides",
"prevention": "Crop rotation"
},
"Tomato___Late_blight": {
"type": "Fungal Disease",
"treatment": "Use fungicide spray",
"prevention": "Remove infected plants"
},
"Tomato___Leaf_Mold": {
"type": "Fungal Disease",
"treatment": "Use fungicide",
"prevention": "Improve ventilation"
},
"Tomato___Septoria_leaf_spot": {
"type": "Fungal Disease",
"treatment": "Apply fungicides",
"prevention": "Avoid leaf wetness"
},
"Tomato___Spider_mites Two-spotted_spider_mite": {
"type": "Pest",
"treatment": "Use insecticidal soap",
"prevention": "Maintain humidity"
},
"Tomato___Target_Spot": {
"type": "Fungal Disease",
"treatment": "Apply fungicides",
"prevention": "Crop rotation"
},
"Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
"type": "Viral Disease",
"treatment": "No cure, remove infected plants",
"prevention": "Control whiteflies"
},
"Tomato___Tomato_mosaic_virus": {
"type": "Viral Disease",
"treatment": "No cure",
"prevention": "Disinfect tools"
},
"Tomato___healthy": {
"type": "Healthy",
"treatment": "No treatment needed",
"prevention": "Maintain plant health"
}
}
# ---------------- PREDICTION ----------------
def model_prediction(test_image):
    from PIL import Image
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))
    input_arr = np.array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    confidence = np.max(prediction) * 100
    return np.argmax(prediction), confidence

# ---------------- PREMIUM CSS WITH GLASS MORPHISM ----------------
st.markdown("""
<style>
/* ====== GLOBAL FONTS ====== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* ====== MAIN BACKGROUND - PREMIUM GRADIENT ====== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2818 0%, #1a3a2a 25%, #0f3820 50%, #1a4d35 75%, #0a2815 100%);
    background-attachment: fixed;
    position: relative;
}

/* ====== ANIMATED GRADIENT OVERLAY ====== */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: 
        radial-gradient(circle at 10% 10%, rgba(34,197,94,0.12), transparent 35%),
        radial-gradient(circle at 90% 90%, rgba(34,197,94,0.08), transparent 40%),
        radial-gradient(ellipse at 50% 50%, rgba(255,255,255,0.02), transparent 60%);
    z-index: 0;
    pointer-events: none;
}

/* ====== CONTENT ABOVE OVERLAY ====== */
.main {
    position: relative;
    z-index: 1;
}

/* ====== SIDEBAR - DARK GLASS ====== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,40,24,0.95), rgba(26,58,42,0.90));
    backdrop-filter: blur(15px);
    padding: 30px 20px;
    border-right: 1px solid rgba(34,197,94,0.15);
    box-shadow: 20px 0 60px rgba(0,0,0,0.4);
}

/* ====== SIDEBAR TITLE ====== */
.sidebar-title {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
    letter-spacing: 0.8px;
    background: linear-gradient(135deg, #22c55e, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sidebar-sub {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 35px;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ====== NAV CONTAINER ====== */
div[role="radiogroup"] {
    gap: 12px;
}

/* ====== NAV ITEM BASE ====== */
div[role="radiogroup"] > label {
    padding: 16px 20px;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1.5px solid rgba(255,255,255,0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    cursor: pointer;
    backdrop-filter: blur(10px);
}

/* ====== NAV TEXT ====== */
div[role="radiogroup"] label span {
    color: #cbd5e1 !important;
    font-size: 16px;
    font-weight: 500;
    letter-spacing: 0.2px;
}

/* ====== NAV HOVER ====== */
div[role="radiogroup"] > label:hover {
    background: rgba(34,197,94,0.08);
    border: 1.5px solid rgba(34,197,94,0.3);
    transform: translateX(8px);
    box-shadow: 0px 8px 20px rgba(34,197,94,0.12);
}

/* ====== NAV ACTIVE ====== */
div[role="radiogroup"] > label[data-selected="true"] {
    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.08));
    border: 1.5px solid rgba(34,197,94,0.5);
    box-shadow: 0px 12px 32px rgba(34,197,94,0.2);
    backdrop-filter: blur(15px);
}

/* ====== NAV ACTIVE TEXT ====== */
div[role="radiogroup"] > label[data-selected="true"] span {
    color: #22c55e !important;
    font-weight: 700;
    text-shadow: 0px 0px 12px rgba(34,197,94,0.4);
}

/* ====== ACTIVE GLOW LINE ====== */
div[role="radiogroup"] > label[data-selected="true"]::before {
    content: "";
    position: absolute;
    left: -8px;
    top: 15%;
    height: 70%;
    width: 5px;
    border-radius: 10px;
    background: linear-gradient(180deg, #22c55e, #10b981);
    box-shadow: 0px 0px 15px #22c55e, 0px 0px 25px rgba(34,197,94,0.5);
}

/* ====== HIDE RADIO INPUT ====== */
div[role="radiogroup"] input {
    display: none;
}

/* ====== H1 STYLING ====== */
h1 {
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}

/* ====== H2 STYLING ====== */
h2 {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
}

/* ====== H3 STYLING ====== */
h3 {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
}

/* ====== TEXT STYLING ====== */
p {
    color: #cbd5e1 !important;
    font-weight: 400 !important;
    line-height: 1.7 !important;
}

/* ====== CARD BASE ====== */
.card {
    background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(34,197,94,0.03));
    padding: 28px;
    border-radius: 20px;
    border: 1.5px solid rgba(34,197,94,0.2);
    box-shadow: 0px 20px 50px rgba(34,197,94,0.1);
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}

.card:hover {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.06));
    border: 1.5px solid rgba(34,197,94,0.35);
    box-shadow: 0px 30px 70px rgba(34,197,94,0.15);
    transform: translateY(-4px);
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("""
<div class="sidebar-title">Dashboard</div>
<div class="sidebar-sub">Plant Analysis System</div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.radio(
    "",
    ["Home", "About", "Disease Recognition"]
)

# ---------------- HOME ----------------
if app_mode == "Home":

    col1, col2 = st.columns([1.3,1])

    with col1:
        st.markdown("""
        <h1>Plant Disease Recognition System</h1>
        <p>
        Detect plant diseases instantly using deep learning. Upload leaf images 
        and get accurate predictions in seconds.
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.image("home_page.jpeg", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class="card">
        <h3>Overview</h3>
        <ul>
        <li>Fast prediction</li>
        <li>High accuracy</li>
        <li>Simple workflow</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card">
        <h3>Steps</h3>
        <ol>
        <li>Go to Predict</li>
        <li>Upload image</li>
        <li>Run prediction</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

# ---------------- ABOUT ----------------
elif(app_mode=="About"):

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- MAIN CARD ----------
    st.markdown("""
    <div style="
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.04));
    padding: 40px;
    border-radius: 24px;
    border: 1.5px solid rgba(34,197,94,0.25);
    box-shadow: 0px 30px 80px rgba(34,197,94,0.15);
    backdrop-filter: blur(20px);
    ">

    <h1 style="color:white; font-size: 40px; margin-bottom: 20px;">About Dataset</h1>

    <p style="color:#cbd5e1; font-size:18px; line-height:1.8; font-weight:500;">
    This dataset has been carefully recreated using offline data augmentation techniques 
    from the original PlantVillage dataset. It consists of approximately <b style="color:#22c55e;">87,000 RGB images</b> 
    of both healthy and diseased plant leaves, categorized into <b style="color:#22c55e;">38 distinct classes</b>. 
    The dataset is structured to ensure high model accuracy and robust performance across 
    different plant species and disease types.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- GRID CARDS ----------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
        padding: 32px;
        border-radius: 20px;
        border: 1.5px solid rgba(34,197,94,0.3);
        box-shadow: 0px 20px 50px rgba(34,197,94,0.12);
        backdrop-filter: blur(15px);
        transition: all 0.3s ease;
        ">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="width: 4px; height: 28px; background: linear-gradient(180deg, #22c55e, #10b981); border-radius: 10px; box-shadow: 0px 0px 12px #22c55e;"></div>
            <h3 style="color:white; margin: 0; font-weight: 700;">Training Data</h3>
        </div>
        <p style="color:#cbd5e1; font-size: 16px; line-height: 1.7; margin: 0;">
        70,295 images used to train the deep learning model, helping it learn patterns 
        of healthy and diseased plant leaves.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
        padding: 32px;
        border-radius: 20px;
        border: 1.5px solid rgba(34,197,94,0.3);
        box-shadow: 0px 20px 50px rgba(34,197,94,0.12);
        backdrop-filter: blur(15px);
        transition: all 0.3s ease;
        ">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="width: 4px; height: 28px; background: linear-gradient(180deg, #22c55e, #10b981); border-radius: 10px; box-shadow: 0px 0px 12px #22c55e;"></div>
            <h3 style="color:white; margin: 0; font-weight: 700;">Validation Data</h3>
        </div>
        <p style="color:#cbd5e1; font-size: 16px; line-height: 1.7; margin: 0;">
        17,572 images used to evaluate model performance during training and prevent overfitting.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
        padding: 32px;
        border-radius: 20px;
        border: 1.5px solid rgba(34,197,94,0.3);
        box-shadow: 0px 20px 50px rgba(34,197,94,0.12);
        backdrop-filter: blur(15px);
        transition: all 0.3s ease;
        ">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="width: 4px; height: 28px; background: linear-gradient(180deg, #22c55e, #10b981); border-radius: 10px; box-shadow: 0px 0px 12px #22c55e;"></div>
            <h3 style="color:white; margin: 0; font-weight: 700;">Test Data</h3>
        </div>
        <p style="color:#cbd5e1; font-size: 16px; line-height: 1.7; margin: 0;">
        33 images used for final predictions to simulate real-world use cases.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- EXTRA INFO ----------
    st.markdown("""
    <div style="
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.04));
    padding: 40px;
    border-radius: 24px;
    border: 1.5px solid rgba(34,197,94,0.25);
    box-shadow: 0px 30px 80px rgba(34,197,94,0.15);
    backdrop-filter: blur(20px);
    ">

    <h2 style="color:white; font-size: 32px; margin-bottom: 28px;">✨ Key Highlights</h2>

    <ul style="color:#cbd5e1; font-size:17px; line-height:2; margin: 0; padding-left: 24px;">
    <li style="margin-bottom: 14px;"><span style="color:#22c55e; font-weight: 700;">◆</span> Dataset includes multiple crop types such as Apple, Tomato, Potato, and more.</li>
    <li style="margin-bottom: 14px;"><span style="color:#22c55e; font-weight: 700;">◆</span> Each class represents a specific disease or healthy condition.</li>
    <li style="margin-bottom: 14px;"><span style="color:#22c55e; font-weight: 700;">◆</span> Images are augmented to improve model generalization.</li>
    <li style="margin-bottom: 14px;"><span style="color:#22c55e; font-weight: 700;">◆</span> Supports real-world prediction scenarios including camera-based input.</li>
    <li><span style="color:#22c55e; font-weight: 700;">◆</span> Optimized for deep learning models like CNN.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)
# ---------------- PREDICT ----------------
elif(app_mode=="Disease Recognition"):

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- HERO ----------
    st.markdown("""
    <div style="text-align:center; padding: 40px 20px;">
        <h1 style="font-size:56px; color:white; font-weight: 800; margin-bottom: 16px; letter-spacing: -0.5px;">
            🔍 Disease Recognition
        </h1>
        <p style="color:#cbd5e1; font-size:20px; font-weight: 500; margin: 0;">
            Upload or capture a plant leaf image and get instant AI-powered analysis
        </p>
        <div style="margin-top: 20px; display: flex; justify-content: center; gap: 8px;">
            <div style="width: 60px; height: 2px; background: linear-gradient(90deg, transparent, #22c55e); border-radius: 10px;"></div>
            <div style="width: 2px; height: 20px; background: #22c55e; border-radius: 10px;"></div>
            <div style="width: 60px; height: 2px; background: linear-gradient(90deg, #22c55e, transparent); border-radius: 10px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- STATS ----------
    col1, col2, col3 = st.columns(3)
    
    col1.markdown("""
    <div style="
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,197,94,0.06));
    padding: 28px;
    border-radius: 18px;
    border: 1.5px solid rgba(34,197,94,0.35);
    text-align: center;
    box-shadow: 0px 15px 40px rgba(34,197,94,0.15);
    ">
        <p style="color:#94a3b8; font-size: 14px; font-weight: 600; margin: 0; letter-spacing: 1px;">ACCURACY</p>
        <h2 style="color:#22c55e; font-size: 48px; font-weight: 800; margin: 12px 0 0 0;">98%</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col2.markdown("""
    <div style="
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,197,94,0.06));
    padding: 28px;
    border-radius: 18px;
    border: 1.5px solid rgba(34,197,94,0.35);
    text-align: center;
    box-shadow: 0px 15px 40px rgba(34,197,94,0.15);
    ">
        <p style="color:#94a3b8; font-size: 14px; font-weight: 600; margin: 0; letter-spacing: 1px;">CLASSES</p>
        <h2 style="color:#22c55e; font-size: 48px; font-weight: 800; margin: 12px 0 0 0;">38</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col3.markdown("""
    <div style="
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,197,94,0.06));
    padding: 28px;
    border-radius: 18px;
    border: 1.5px solid rgba(34,197,94,0.35);
    text-align: center;
    box-shadow: 0px 15px 40px rgba(34,197,94,0.15);
    ">
        <p style="color:#94a3b8; font-size: 14px; font-weight: 600; margin: 0; letter-spacing: 1px;">DATASET</p>
        <h2 style="color:#22c55e; font-size: 48px; font-weight: 800; margin: 12px 0 0 0;">87K+</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- FEATURE CHIPS ----------
    st.markdown("""
    <div style="text-align:center; display: flex; justify-content: center; flex-wrap: wrap; gap: 12px;">
        <span style="background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.08)); padding:12px 24px; border-radius:24px; border: 1.5px solid rgba(34,197,94,0.3); color:#22c55e; font-weight: 700; font-size: 15px; box-shadow: 0px 8px 20px rgba(34,197,94,0.15); backdrop-filter: blur(10px);">⚡ Real-time Detection</span>
        <span style="background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.08)); padding:12px 24px; border-radius:24px; border: 1.5px solid rgba(34,197,94,0.3); color:#22c55e; font-weight: 700; font-size: 15px; box-shadow: 0px 8px 20px rgba(34,197,94,0.15); backdrop-filter: blur(10px);">🤖 AI Powered</span>
        <span style="background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.08)); padding:12px 24px; border-radius:24px; border: 1.5px solid rgba(34,197,94,0.3); color:#22c55e; font-weight: 700; font-size: 15px; box-shadow: 0px 8px 20px rgba(34,197,94,0.15); backdrop-filter: blur(10px);">✓ 98% Accuracy</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------- MAIN CARD ----------
    st.markdown("""
    <div style="
    max-width:900px;
    margin:auto;
    background:rgba(255,255,255,0.04);
    padding:35px;
    border-radius:24px;
    backdrop-filter:blur(20px);
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0px 20px 60px rgba(0,0,0,0.5);
    ">
    """, unsafe_allow_html=True)

    # ---------- STEPS VISUAL ----------
    col1, col2, col3 = st.columns(3)

    col1.markdown("### Upload\nUpload leaf image")
    col2.markdown("### Scan\nUse camera")
    col3.markdown("### Detect\nGet results")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- INPUT ----------
    st.markdown("<p style='color:#d1d5db;'>Select Input Method</p>", unsafe_allow_html=True)

    option = st.radio("", ["Upload Image", "Use Camera"], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)

    test_image = None

    # ---------- UPLOAD ----------
    if option == "Upload Image":
        st.markdown("""
        <div style="
        border:2.5px dashed rgba(34,197,94,0.4);
        padding:60px 40px;
        border-radius:24px;
        text-align:center;
        color:#cbd5e1;
        background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(34,197,94,0.02));
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        font-weight: 600;
        font-size: 18px;
        ">
        📁 Drag & drop or click to upload leaf image
        </div>
        """, unsafe_allow_html=True)

        test_image = st.file_uploader("", label_visibility="collapsed")

    # ---------- CAMERA ----------
    else:
        test_image = st.camera_input("Capture leaf image")

    # ---------- IMAGE ----------
    if test_image is not None:
        st.image(test_image, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- BUTTON STYLE ----------
    st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 56px;
        border-radius: 16px;
        background: linear-gradient(135deg, #22c55e, #10b981);
        font-size: 18px;
        font-weight: 700;
        border: 1.5px solid rgba(34,197,94,0.4);
        box-shadow: 0px 15px 40px rgba(34,197,94,0.3), 0px 0px 20px rgba(34,197,94,0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: white;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0px 20px 50px rgba(34,197,94,0.4), 0px 0px 30px rgba(34,197,94,0.3);
    }
    .stButton>button:active {
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- BUTTON ----------
    if st.button("Analyze Plant"):

        if test_image is not None:

            with st.spinner("Analyzing..."):

                if not is_leaf(test_image):
                    st.error("🚫 No plant leaf detected. Please upload or capture a clear plant leaf image.")
                else:
                    result_index, confidence = model_prediction(test_image)

                    class_name = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
                    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
                    'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
                    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
                    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
                    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch','Strawberry___healthy',
                    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
                    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']

                    st.markdown(f"""
                    <div style="
                    margin-top: 35px;
                    padding: 35px;
                    border-radius: 22px;
                    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.06));
                    border: 2px solid rgba(34,197,94,0.4);
                    box-shadow: 0px 20px 60px rgba(34,197,94,0.25), inset 0px 1px 0px rgba(255,255,255,0.1);
                    backdrop-filter: blur(15px);
                    animation: slideUp 0.5s ease-out;
                    ">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                        <div style="width: 6px; height: 32px; background: linear-gradient(180deg, #22c55e, #10b981); border-radius: 10px; box-shadow: 0px 0px 15px #22c55e;"></div>
                        <h3 style="color:#22c55e; margin: 0; font-size: 18px; font-weight: 700; letter-spacing: 0.5px;">✓ PREDICTION RESULT</h3>
                    </div>
                    <p style="color:white; font-size: 26px; font-weight: 700; margin: 16px 0 0 0; letter-spacing: 0.3px;">
                    {class_name[result_index]}
                    </p>
                    </div>
                    <style>
                    @keyframes slideUp {{
                        from {{ opacity: 0; transform: translateY(20px); }}
                        to {{ opacity: 1; transform: translateY(0); }}
                    }}
                    </style>
                    """, unsafe_allow_html=True)

                    predicted_label = class_name[result_index].strip()

                    st.markdown(f"""
<div style="
margin-top: 20px;
padding: 28px;
border-radius: 18px;
background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
border: 1.5px solid rgba(34,197,94,0.3);
box-shadow: 0px 15px 40px rgba(34,197,94,0.12);
backdrop-filter: blur(15px);
">
<p style="color:#94a3b8; font-size: 14px; font-weight: 600; margin: 0; letter-spacing: 1px;">CONFIDENCE SCORE</p>
<div style="display: flex; align-items: center; gap: 16px; margin-top: 14px;">
    <h2 style="color:#22c55e; font-size: 44px; font-weight: 800; margin: 0;">{confidence:.1f}%</h2>
    <div style="flex: 1;">
        <div style="background: rgba(255,255,255,0.08); height: 8px; border-radius: 10px; overflow: hidden;">
            <div style="width: {confidence}%; height: 100%; background: linear-gradient(90deg, #22c55e, #10b981); border-radius: 10px; box-shadow: 0px 0px 15px #22c55e; transition: width 0.6s ease-out;"></div>
        </div>
    </div>
</div>
</div>
""", unsafe_allow_html=True)

                    if predicted_label in disease_info:
                        disease_data = disease_info[predicted_label]
                        disease_type = disease_data['type']

                        st.markdown(f"""
<div style="
margin-top: 30px;
padding: 35px;
border-radius: 22px;
background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.04));
border: 1.5px solid rgba(34,197,94,0.25);
box-shadow: 0px 25px 70px rgba(34,197,94,0.15);
backdrop-filter: blur(20px);
">
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 28px;">
    <div style="width: 6px; height: 32px; background: linear-gradient(180deg, #22c55e, #10b981); border-radius: 10px; box-shadow: 0px 0px 15px #22c55e;"></div>
    <h3 style="color:#22c55e; margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0.5px;">📋 DISEASE INFORMATION</h3>
</div>

<div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 14px; border-left: 3px solid #22c55e; margin-bottom: 20px;">
    <p style="color:#94a3b8; font-size: 13px; margin: 0; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;">Disease Type</p>
    <p style="color:white; font-size: 20px; font-weight: 700; margin: 8px 0 0 0;">{disease_type}</p>
</div>

<div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 14px; border-left: 3px solid #22c55e; margin-bottom: 20px;">
    <p style="color:#94a3b8; font-size: 13px; margin: 0; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;">Treatment</p>
    <p style="color:#cbd5e1; font-size: 16px; margin: 8px 0 0 0; line-height: 1.6; font-weight: 500;">{disease_data['treatment']}</p>
</div>

<div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 14px; border-left: 3px solid #22c55e;">
    <p style="color:#94a3b8; font-size: 13px; margin: 0; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;">Prevention</p>
    <p style="color:#cbd5e1; font-size: 16px; margin: 8px 0 0 0; line-height: 1.6; font-weight: 500;">{disease_data['prevention']}</p>
</div>
</div>
""", unsafe_allow_html=True)

        else:
            st.warning("Upload or capture image first.")

    st.markdown("</div>", unsafe_allow_html=True)