#!/usr/bin/env python3
"""
CivicGuard Streamlit Dashboard
A beautiful, real-time harassment detection monitoring interface.
Runs python src/harassment_detection.py when Start is clicked.
"""
import streamlit as st
import subprocess
import time
import os
import signal
import sys
from datetime import datetime
import plotly.graph_objects as go

# Add src to path for config imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from config import load_alerts, clear_alerts, update_detection_status

# ==================== CONFIGURATION ====================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="CivicGuard - AI Harassment Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main theme - Dark mode with gradient accents */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    
    .main-header h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #a0aec0;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, rgba(45, 55, 72, 0.8), rgba(26, 32, 44, 0.9));
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Status indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-running {
        background: linear-gradient(90deg, #48bb78, #38a169);
        color: white;
    }
    
    .status-stopped {
        background: linear-gradient(90deg, #fc8181, #e53e3e);
        color: white;
    }
    
    /* Threat meter */
    .threat-meter-container {
        background: linear-gradient(145deg, rgba(45, 55, 72, 0.9), rgba(26, 32, 44, 0.95));
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        text-align: center;
    }
    
    .threat-level {
        font-size: 5rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    
    .threat-safe { color: #48bb78; }
    .threat-low { color: #68d391; }
    .threat-medium { color: #f6e05e; }
    .threat-high { color: #f6ad55; }
    .threat-critical { color: #fc8181; }
    
    /* Progress bar override */
    .stProgress > div > div > div {
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Alert history cards */
    .alert-card {
        background: rgba(45, 55, 72, 0.6);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .alert-critical { border-left-color: #e53e3e; }
    .alert-high { border-left-color: #ed8936; }
    .alert-medium { border-left-color: #ecc94b; }
    .alert-low { border-left-color: #48bb78; }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(145deg, rgba(66, 153, 225, 0.2), rgba(49, 130, 206, 0.1));
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(66, 153, 225, 0.3);
        margin: 1rem 0;
    }
    
    .feature-card {
        background: linear-gradient(145deg, rgba(45, 55, 72, 0.7), rgba(26, 32, 44, 0.8));
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    
    .feature-card h4 {
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 32, 44, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.8);
    }
</style>
""", unsafe_allow_html=True)


# ==================== PROCESS MANAGEMENT ====================
def start_detection_process(video_source=0):
    """Start the harassment detection script."""
    try:
        # Build the command
        script_path = os.path.join(PROJECT_DIR, "src", "harassment_detection.py")
        src_path = os.path.join(PROJECT_DIR, "src")
        
        if isinstance(video_source, int) or (isinstance(video_source, str) and video_source.isdigit()):
            # Webcam source
            cmd = ["python3", script_path]
        else:
            # Video file path
            cmd = ["python3", script_path, str(video_source)]
        
        # Set up environment with PYTHONPATH to ensure correct imports
        env = os.environ.copy()
        env["PYTHONPATH"] = src_path + ":" + env.get("PYTHONPATH", "")
        
        # Start the process WITHOUT capturing output
        # This allows the OpenCV window to open properly
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_DIR,
            env=env,
            # Don't capture stdout/stderr - let it display in terminal
            # This is critical for OpenCV windows to work
        )
        
        return process
    except Exception as e:
        st.error(f"Error starting detection: {e}")
        return None


def stop_detection_process(process):
    """Stop the detection process."""
    try:
        if process and process.poll() is None:
            # Try graceful termination first
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't respond
                process.kill()
                process.wait()
        return True
    except Exception as e:
        return False


def is_process_running(process):
    """Check if the process is still running."""
    if process is None:
        return False
    return process.poll() is None


# ==================== UI COMPONENTS ====================
def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ CivicGuard</h1>
        <p>AI-Powered Real-Time Harassment Detection</p>
    </div>
    """, unsafe_allow_html=True)


def render_control_buttons():
    """Render start/stop control buttons."""
    is_running = st.session_state.get("is_running", False)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_clicked = st.button(
            "▶️ Start", 
            type="primary", 
            disabled=is_running, 
            use_container_width=True,
            key="start_btn"
        )
        
        if start_clicked and not is_running:
            video_source = st.session_state.get("video_source", 0)
            process = start_detection_process(video_source)
            
            if process:
                st.session_state.detection_process = process
                st.session_state.is_running = True
                st.session_state.start_time = datetime.now()
                st.success("✅ Detection started! Camera window should open.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to start detection process.")
    
    with col2:
        stop_clicked = st.button(
            "⏹️ Stop", 
            type="secondary", 
            disabled=not is_running, 
            use_container_width=True,
            key="stop_btn"
        )
        
        if stop_clicked and is_running:
            process = st.session_state.get("detection_process")
            if stop_detection_process(process):
                st.session_state.is_running = False
                st.session_state.detection_process = None
                st.info("⏹️ Detection stopped.")
                st.rerun()
            else:
                st.error("❌ Failed to stop detection process.")
    
    with col3:
        # Check if process is still running
        process = st.session_state.get("detection_process")
        actual_running = is_process_running(process)
        
        # Sync state if process died
        if st.session_state.get("is_running") and not actual_running:
            st.session_state.is_running = False
            st.session_state.detection_process = None
        
        if st.session_state.get("is_running", False):
            st.markdown(
                '<span class="status-badge status-running">🟢 RUNNING</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-badge status-stopped">🔴 STOPPED</span>',
                unsafe_allow_html=True
            )
    
    return st.session_state.get("is_running", False)


def render_config_bar():
    """Render the configuration bar."""
    alert_interval = st.session_state.get("alert_interval", 3)
    confidence_threshold = st.session_state.get("confidence_threshold", 0.75)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"⏱️ **Alert Interval:** {alert_interval}s")
    
    with col2:
        st.markdown(f"🎯 **Confidence Threshold:** {confidence_threshold*100:.0f}%")
    
    with col3:
        st.markdown("📊 **Mode:** Aggregated Threat Detection")


def render_threat_meter():
    """Render the main threat meter with live alert data."""
    st.markdown("### 🚨 Threat Meter")
    
    is_running = st.session_state.get("is_running", False)
    
    # Load alerts from shared file
    alerts_data = load_alerts()
    is_threat_active = alerts_data.get("is_threat_active", False)
    current_threat = alerts_data.get("current_threat")
    alerts_list = alerts_data.get("alerts", [])
    camera_source = alerts_data.get("camera_source", "Unknown")
    
    if is_running:
        start_time = st.session_state.get("start_time", datetime.now())
        elapsed = datetime.now() - start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        
        # Check if there's an active threat
        if is_threat_active and current_threat:
            risk_score = current_threat.get("risk_score", 0)
            risk_percent = int(risk_score * 100)
            source = current_threat.get("source", camera_source)
            timestamp = current_threat.get("timestamp", "Unknown")
            
            # Determine threat level styling
            if risk_percent >= 90:
                threat_class = "threat-critical"
                level_text = "CRITICAL"
            elif risk_percent >= 75:
                threat_class = "threat-high"
                level_text = "HIGH"
            elif risk_percent >= 50:
                threat_class = "threat-medium"
                level_text = "MEDIUM"
            else:
                threat_class = "threat-low"
                level_text = "LOW"
            
            st.markdown(f"""
            <div class="threat-meter-container" style="border: 2px solid #fc8181; animation: pulse 1s infinite;">
                <h4 style="color: #fc8181; margin-bottom: 0;">⚠️ HARASSMENT DETECTED!</h4>
                <div class="threat-level {threat_class}">{risk_percent}%</div>
                <p style="color: #fc8181; font-size: 1.2rem; margin: 0.5rem 0;">
                    Level: <strong>{level_text}</strong>
                </p>
                <div style="background: rgba(252, 129, 129, 0.2); border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                    <p style="color: #fff; margin: 0.3rem 0;">📹 <strong>Source:</strong> {source}</p>
                    <p style="color: #fff; margin: 0.3rem 0;">🕐 <strong>Detected at:</strong> {timestamp}</p>
                    <p style="color: #fff; margin: 0.3rem 0;">👤 <strong>Person ID:</strong> {current_threat.get('person_id', 'N/A')}</p>
                </div>
                <p style="color: #a0aec0; margin-top: 0.5rem;">
                    ⏱️ Running for: {elapsed_str}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add pulsing animation CSS
            st.markdown("""
            <style>
                @keyframes pulse {
                    0% { box-shadow: 0 0 0 0 rgba(252, 129, 129, 0.4); }
                    70% { box-shadow: 0 0 0 15px rgba(252, 129, 129, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(252, 129, 129, 0); }
                }
            </style>
            """, unsafe_allow_html=True)
            
            st.error(f"🚨 **ALERT:** Harassment pattern detected from **{source}**! Check the camera window for visual confirmation.")
            
        else:
            # Running but no current threat
            st.markdown(f"""
            <div class="threat-meter-container">
                <h4 style="color: #a0aec0; margin-bottom: 0;">🎥 Detection Active</h4>
                <div class="threat-level threat-safe" style="font-size: 3rem;">MONITORING</div>
                <p style="color: #48bb78; font-size: 1.2rem; margin: 0.5rem 0;">
                    Source: <strong>{camera_source}</strong>
                </p>
                <p style="color: #a0aec0; margin-top: 1rem;">
                    ⏱️ Running for: {elapsed_str}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("💡 **Tip:** The detection runs in a separate OpenCV window. Press 'q' in that window to stop.")
        
    else:
        st.markdown(f"""
        <div class="threat-meter-container">
            <h4 style="color: #a0aec0; margin-bottom: 0;">Threat Meter</h4>
            <div class="threat-level threat-safe">0%</div>
            <p style="color: #48bb78; font-size: 1.2rem; margin: 0.5rem 0;">
                Level: <strong>SAFE</strong>
            </p>
            <p style="color: #a0aec0; margin-top: 1rem;">
                ✅ System ready - Click Start to begin
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(0.0)


def render_system_status():
    """Render system status information and recent alerts."""
    st.markdown("### 📊 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    is_running = st.session_state.get("is_running", False)
    
    # Load alerts data for source and count info
    alerts_data = load_alerts()
    camera_source = alerts_data.get("camera_source", "Camera 0")
    alerts_list = alerts_data.get("alerts", [])
    alerts_count = len(alerts_list)
    
    with col1:
        st.metric(
            label="🎥 Camera",
            value="Active" if is_running else "Idle"
        )
    
    with col2:
        st.metric(
            label="📹 Source",
            value=camera_source if camera_source else "Camera 0"
        )
    
    with col3:
        if is_running:
            start_time = st.session_state.get("start_time", datetime.now())
            elapsed = datetime.now() - start_time
            st.metric(
                label="⏱️ Uptime",
                value=str(elapsed).split('.')[0]
            )
        else:
            st.metric(
                label="⏱️ Uptime",
                value="--:--:--"
            )
    
    with col4:
        st.metric(
            label="🚨 Total Alerts",
            value=alerts_count
        )
    
    # Show recent alerts history
    if alerts_count > 0:
        st.markdown("### 📋 Recent Alerts")
        
        # Show last 5 alerts in reverse order (newest first)
        for alert in reversed(alerts_list[-5:]):
            risk_score = alert.get("risk_score", 0)
            risk_percent = int(risk_score * 100)
            source = alert.get("source", "Unknown")
            timestamp = alert.get("timestamp", "Unknown")
            person_id = alert.get("person_id", "N/A")
            
            # Determine alert severity class
            if risk_percent >= 75:
                alert_class = "alert-critical"
                severity_emoji = "🔴"
            elif risk_percent >= 50:
                alert_class = "alert-high"
                severity_emoji = "🟠"
            else:
                alert_class = "alert-medium"
                severity_emoji = "🟡"
            
            st.markdown(f"""
            <div class="alert-card {alert_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #fff;">{severity_emoji} <strong>{risk_percent}% Risk</strong> - Person #{person_id}</span>
                    <span style="color: #a0aec0; font-size: 0.8rem;">{timestamp}</span>
                </div>
                <p style="color: #cbd5e0; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                    📹 Source: <strong>{source}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)


def render_features():
    """Render feature highlights."""
    st.markdown("### ✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Real-Time Detection</h4>
            <p>AI-powered harassment detection using YOLOv8 pose estimation and XGBoost classification with 85% accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>👥 Interaction Analysis</h4>
            <p>Monitors interpersonal distance, body orientation, and temporal behavior patterns between individuals.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🔒 Privacy-First</h4>
            <p>Pose-based analysis with automatic face blurring. No facial recognition or personal identification.</p>
        </div>
        """, unsafe_allow_html=True)


def render_how_it_works():
    """Render how it works section."""
    st.markdown("### 🔍 How It Works")
    
    st.markdown("""
    <div class="info-box">
        <ol style="color: #e2e8f0; margin: 0; padding-left: 1.5rem;">
            <li><strong>Click Start</strong> — Opens a camera window with live detection</li>
            <li><strong>Monitor</strong> — Watch the camera feed for real-time analysis</li>
            <li><strong>Alerts</strong> — Visual alerts appear when harassment patterns are detected</li>
            <li><strong>Press 'q'</strong> — In the camera window to stop detection</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar configuration."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        st.markdown("---")
        
        # Video Source Selection
        st.markdown("### 🎥 Video Source")
        
        source_options = ["📹 Webcam (Camera 0)", "🎬 Sample Video (samplevideo.mp4)", "📁 Custom Video File"]
        source_choice = st.radio(
            "Select input source:",
            source_options,
            index=st.session_state.get("source_index", 0),
            disabled=st.session_state.get("is_running", False),
            key="source_radio"
        )
        
        # Handle source selection
        if source_choice == source_options[0]:  # Webcam
            st.session_state.source_index = 0
            camera_idx = st.number_input(
                "Camera Index",
                min_value=0,
                max_value=10,
                value=0,
                help="Camera device index (0 for default webcam)",
                disabled=st.session_state.get("is_running", False)
            )
            st.session_state.video_source = int(camera_idx)
            st.info(f"📹 Using webcam device {camera_idx}")
            
        elif source_choice == source_options[1]:  # Sample video
            st.session_state.source_index = 1
            sample_path = os.path.join(PROJECT_DIR, "samplevideo.mp4")
            if os.path.exists(sample_path):
                st.session_state.video_source = sample_path
                st.success("✅ Sample video ready!")
                st.caption(f"📍 {sample_path}")
            else:
                st.error("❌ samplevideo.mp4 not found in project root")
                st.session_state.video_source = 0
                
        else:  # Custom video file
            st.session_state.source_index = 2
            video_path = st.text_input(
                "Video file path:",
                value=st.session_state.get("custom_video_path", ""),
                help="Enter the full path to your video file",
                disabled=st.session_state.get("is_running", False)
            )
            st.session_state.custom_video_path = video_path
            
            if video_path:
                if os.path.exists(video_path):
                    st.session_state.video_source = video_path
                    st.success("✅ Video file found!")
                else:
                    st.error("❌ File not found")
                    st.session_state.video_source = 0
            else:
                st.session_state.video_source = 0
                st.info("Enter a video file path above")
        
        st.markdown("---")
        
        # Alert interval (info only - controlled by script)
        st.markdown("### ⚙️ Detection Settings")
        st.markdown("⏱️ **Alert Cooldown:** 20 seconds")
        st.markdown("🎯 **Confidence Threshold:** 75%")
        st.caption("(These are configured in the detection script)")
        
        st.markdown("---")
        
        # Detection Status
        st.markdown("### 🎥 Detection Status")
        
        # Load alerts data
        alerts_data = load_alerts()
        alerts_count = len(alerts_data.get("alerts", []))
        current_source = alerts_data.get("camera_source", None)
        
        if st.session_state.get("is_running", False):
            st.success("✅ Detection Running")
            process = st.session_state.get("detection_process")
            if process:
                st.info(f"PID: {process.pid}")
            if current_source:
                st.info(f"📹 Source: {current_source}")
        else:
            st.info("⏸️ Detection Stopped")
        
        # Show alert count
        if alerts_count > 0:
            st.warning(f"🚨 Total Alerts: {alerts_count}")
            
            # Clear alerts button
            if st.button("🗑️ Clear Alerts", use_container_width=True):
                clear_alerts()
                st.success("✅ Alerts cleared!")
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; color: #718096; font-size: 0.8rem;">
            <p>🛡️ CivicGuard v1.0</p>
            <p>AI-Powered Harassment Detection</p>
            <p style="margin-top: 1rem;">
                <small>Using YOLOv8 + XGBoost</small>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==================== MAIN APP ====================
def main():
    """Main application entry point."""
    # Initialize session state
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "detection_process" not in st.session_state:
        st.session_state.detection_process = None
    if "video_source" not in st.session_state:
        st.session_state.video_source = 0
    if "source_index" not in st.session_state:
        st.session_state.source_index = 0
    if "custom_video_path" not in st.session_state:
        st.session_state.custom_video_path = ""
    if "alert_interval" not in st.session_state:
        st.session_state.alert_interval = 20
    if "confidence_threshold" not in st.session_state:
        st.session_state.confidence_threshold = 0.75
    
    # Render sidebar
    render_sidebar()
    
    # Render header
    render_header()
    
    # Render control buttons and get status
    is_running = render_control_buttons()
    
    st.markdown("---")
    
    # Render config bar
    render_config_bar()
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Threat Meter
        render_threat_meter()
        
        st.markdown("---")
        
        # System Status
        render_system_status()
    
    with col2:
        # How it works
        render_how_it_works()
        
        st.markdown("---")
        
        # Live output (if running)
        if is_running:
            st.markdown("### 📋 Detection Log")
            st.markdown("""
            <div class="info-box">
                <p style="color: #a0aec0;">
                    Detection output is shown in the <strong>camera window</strong> 
                    and the <strong>terminal</strong> where you started the dashboard.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features section
    render_features()
    
    # Auto-refresh when running to update uptime
    if is_running:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
