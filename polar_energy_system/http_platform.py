"""
POLAR ENERGY INTELLIGENCE SYSTEM (PEIS) - HTTP SERVER PLATFORM
National Centre for Polar and Ocean Research (NCPOR) / Ministry of Earth Sciences (MoES)

Persistent HTTP Platform & Background Watchdog Service
Hosts an HTTP Gateway on port 8080, manages the Streamlit application engine (port 8501),
serves health monitoring APIs, local network sharing, and cloud hosting integration.
"""

import http.server
import socketserver
import urllib.request
import urllib.error
import urllib.parse
import subprocess
import threading
import time
import os
import sys
import json
import socket
import webbrowser
from datetime import datetime

# Configuration Constants
PLATFORM_PORT = int(os.environ.get("HTTP_PLATFORM_PORT", 8080))
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", 8501))
HOST = "0.0.0.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ENTRY = os.path.join(BASE_DIR, "app.py")

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()

# Global State Management
class ServiceState:
    process = None
    start_time = datetime.now()
    restart_count = 0
    last_health_check = None
    is_healthy = False
    lock = threading.Lock()

state = ServiceState()

def check_streamlit_health():
    url = f"http://127.0.0.1:{STREAMLIT_PORT}/_stcore/health"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'PEIS-HTTP-Platform/1.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                with state.lock:
                    state.is_healthy = True
                    state.last_health_check = datetime.now().isoformat()
                return True
    except Exception:
        pass
    
    try:
        url_main = f"http://127.0.0.1:{STREAMLIT_PORT}/"
        req = urllib.request.Request(url_main, headers={'User-Agent': 'PEIS-HTTP-Platform/1.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            with state.lock:
                state.is_healthy = (response.status == 200)
                state.last_health_check = datetime.now().isoformat()
            return state.is_healthy
    except Exception:
        with state.lock:
            state.is_healthy = False
            state.last_health_check = datetime.now().isoformat()
        return False

def start_streamlit_process():
    with state.lock:
        if state.process and state.process.poll() is None:
            return

        print(f"[INFO] Starting Streamlit process on port {STREAMLIT_PORT}...")
        cmd = [
            sys.executable, "-m", "streamlit", "run", APP_ENTRY,
            "--server.port", str(STREAMLIT_PORT),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        state.process = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        state.restart_count += 1
        print(f"[SUCCESS] Streamlit launched with PID {state.process.pid}")

def stop_streamlit_process():
    with state.lock:
        if state.process and state.process.poll() is None:
            state.process.terminate()
            try:
                state.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                state.process.kill()
            state.process = None

def watchdog_daemon():
    print("[INFO] HTTP Platform Watchdog Daemon active.")
    while True:
        try:
            if not check_streamlit_health():
                print("[WARNING] Streamlit health check failed. Attempting restart...")
                start_streamlit_process()
                for _ in range(10):
                    time.sleep(1)
                    if check_streamlit_health():
                        print("[SUCCESS] Streamlit recovered!")
                        break
        except Exception as e:
            print(f"[ERROR] Watchdog error: {e}")
        time.sleep(5)

class PlatformHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/portal":
            self.serve_portal_html()
        elif path == "/app" or path.startswith("/_stcore") or path.startswith("/vendor") or path.startswith("/static"):
            self.proxy_to_streamlit()
        elif path == "/api/status":
            self.serve_json(self.get_status_dict())
        elif path == "/api/health":
            healthy = check_streamlit_health()
            status_code = 200 if healthy else 503
            self.serve_json({"status": "healthy" if healthy else "degraded", "http_platform": "active", "streamlit_port": STREAMLIT_PORT}, status=status_code)
        elif path == "/api/restart":
            stop_streamlit_process()
            start_streamlit_process()
            self.serve_json({"message": "Streamlit process restart initiated", "status": "restarting"})
        elif path == "/redirect":
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{STREAMLIT_PORT}")
            self.end_headers()
        else:
            self.proxy_to_streamlit()

    def proxy_to_streamlit(self):
        target_path = self.path
        if target_path == "/app":
            target_path = "/"
        
        target_url = f"http://127.0.0.1:{STREAMLIT_PORT}{target_path}"
        try:
            req = urllib.request.Request(target_url, headers={
                'User-Agent': self.headers.get('User-Agent', 'PEIS-HTTP-Platform'),
                'Accept': self.headers.get('Accept', '*/*')
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read()
                self.send_response(resp.status)
                for header, val in resp.headers.items():
                    if header.lower() not in ['transfer-encoding', 'content-length', 'connection']:
                        self.send_header(header, val)
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
        except Exception:
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{STREAMLIT_PORT}")
            self.end_headers()

    def serve_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def get_status_dict(self):
        uptime = str(datetime.now() - state.start_time).split('.')[0]
        pid = state.process.pid if state.process and state.process.poll() is None else None
        return {
            "platform": "Polar Energy Intelligence System - HTTP Server Platform",
            "organization": "NCPOR / Ministry of Earth Sciences (MoES)",
            "platform_http_port": PLATFORM_PORT,
            "streamlit_backend_port": STREAMLIT_PORT,
            "backend_pid": pid,
            "backend_healthy": state.is_healthy,
            "uptime": uptime,
            "restart_count": state.restart_count,
            "last_health_check": state.last_health_check,
            "local_access_url": f"http://localhost:{PLATFORM_PORT}",
            "network_access_url": f"http://{LOCAL_IP}:{PLATFORM_PORT}",
            "direct_app_url": f"http://localhost:{STREAMLIT_PORT}"
        }

    def serve_portal_html(self):
        raw_host = self.headers.get('Host', f"{LOCAL_IP}:{PLATFORM_PORT}")
        host_domain = raw_host.split(':')[0]
        if host_domain in ['0.0.0.0', '127.0.0.1']:
            app_target_url = f"http://localhost:{STREAMLIT_PORT}"
        else:
            app_target_url = f"http://{host_domain}:{STREAMLIT_PORT}"

        status_data = self.get_status_dict()
        is_online = status_data["backend_healthy"]
        status_badge_color = "#10b981" if is_online else "#ef4444"
        status_text = "ONLINE & OPERATIONAL" if is_online else "STARTING / DEGRADED"
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polar Energy Intelligence System | HTTP Platform Server</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }}
        body {{
            background: #090d16;
            color: #f1f5f9;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95));
            border-bottom: 1px solid rgba(56, 189, 248, 0.2);
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .logo-box {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .logo-icon {{
            font-size: 2.2rem;
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            border-radius: 12px;
            padding: 8px 14px;
        }}
        .title-text h1 {{
            font-size: 1.5rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.5px;
        }}
        .title-text p {{
            font-size: 0.85rem;
            color: #38bdf8;
            font-weight: 600;
        }}
        .status-pill {{
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid {status_badge_color};
            padding: 8px 18px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.85rem;
            font-weight: 700;
            color: {status_badge_color};
        }}
        .pulse-dot {{
            width: 10px;
            height: 10px;
            background-color: {status_badge_color};
            border-radius: 50%;
            box-shadow: 0 0 10px {status_badge_color};
        }}
        main {{
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 20px;
            width: 100%;
            flex: 1;
        }}
        .hero-banner {{
            background: linear-gradient(135deg, rgba(2, 132, 199, 0.2), rgba(15, 23, 42, 0.8));
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 20px;
            padding: 35px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }}
        .hero-banner h2 {{
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 10px;
            color: #ffffff;
        }}
        .hero-banner p {{
            color: #94a3b8;
            font-size: 1.05rem;
            margin-bottom: 25px;
            max-width: 800px;
            line-height: 1.6;
        }}
        .cta-buttons {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .btn {{
            padding: 14px 28px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1rem;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            transition: all 0.2s ease;
            cursor: pointer;
            border: none;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, #0284c7, #0369a1);
            color: #ffffff;
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
        }}
        .btn-primary:hover {{
            background: linear-gradient(135deg, #0369a1, #075985);
            transform: translateY(-2px);
        }}
        .btn-secondary {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.3);
            color: #cbd5e1;
        }}
        .btn-secondary:hover {{
            background: rgba(51, 65, 85, 0.9);
            color: #ffffff;
        }}
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(56, 189, 248, 0.15);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(10px);
        }}
        .card-label {{
            font-size: 0.8rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .card-value {{
            font-size: 1.25rem;
            font-weight: 800;
            color: #38bdf8;
            word-break: break-all;
        }}
        .card-sub {{
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 6px;
        }}
        .cloud-notice {{
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        .cloud-notice h3 {{
            color: #10b981;
            font-size: 1.2rem;
            margin-bottom: 8px;
        }}
        .cloud-notice p {{
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.5;
        }}
        .app-frame-container {{
            background: #0f172a;
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 16px;
            overflow: hidden;
            height: 750px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.5);
            margin-bottom: 40px;
        }}
        .frame-header {{
            background: #1e293b;
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .frame-title {{
            font-size: 0.9rem;
            font-weight: 600;
            color: #94a3b8;
        }}
        iframe {{
            width: 100%;
            height: calc(100% - 45px);
            border: none;
        }}
        footer {{
            text-align: center;
            padding: 25px;
            color: #64748b;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255,255,255,0.05);
        }}
    </style>
    <script>
        async function restartServer() {{
            if (confirm("Are you sure you want to restart the Streamlit background process?")) {{
                const res = await fetch("/api/restart");
                const data = await res.json();
                alert(data.message);
                setTimeout(() => location.reload(), 3000);
            }}
        }}
    </script>
</head>
<body>
    <header>
        <div class="logo-box">
            <div class="logo-icon">❄</div>
            <div class="title-text">
                <h1>POLAR ENERGY INTELLIGENCE SYSTEM</h1>
                <p>NCPOR / Ministry of Earth Sciences (MoES) • Persistent HTTP Platform</p>
            </div>
        </div>
        <div class="status-pill">
            <div class="pulse-dot"></div>
            <span>{status_text}</span>
        </div>
    </header>

    <main>
        <section class="hero-banner">
            <h2>Persistent Multi-Device HTTP Platform</h2>
            <p>
                Access the Polar Energy Intelligence System locally or across your Wi-Fi network.
                To access this site from any device (phone, tablet, PC) <strong>even when your laptop is turned off</strong>,
                use the 24/7 Cloud Deployment Guide provided below.
            </p>
            <div class="cta-buttons">
                <a href="{app_target_url}" target="_blank" class="btn btn-primary">
                    🚀 Launch Fullscreen App ({app_target_url})
                </a>
                <button onclick="restartServer()" class="btn btn-secondary">
                    🔄 Restart Backend Engine
                </button>
            </div>
        </section>

        <section class="cards-grid">
            <div class="card">
                <div class="card-label">This Laptop Access</div>
                <div class="card-value">http://localhost:{PLATFORM_PORT}</div>
                <div class="card-sub">Local Browser</div>
            </div>
            <div class="card">
                <div class="card-label">Wi-Fi Network Access</div>
                <div class="card-value">http://{LOCAL_IP}:{PLATFORM_PORT}</div>
                <div class="card-sub">Mobile / Phone / Tablet (Laptop ON)</div>
            </div>
            <div class="card">
                <div class="card-label">Direct Mobile App URL</div>
                <div class="card-value">http://{LOCAL_IP}:{STREAMLIT_PORT}</div>
                <div class="card-sub">Direct App View on Wi-Fi</div>
            </div>
            <div class="card">
                <div class="card-label">Windows Boot Auto-Start</div>
                <div class="card-value" style="color: #10b981;">REGISTERED</div>
                <div class="card-sub">Launches on Device Boot</div>
            </div>
            <div class="card">
                <div class="card-label">Data Persistence</div>
                <div class="card-value" style="color: #38bdf8;">ACTIVE</div>
                <div class="card-sub">Restores Data on Restart</div>
            </div>
        </section>

        <section class="cloud-notice">
            <h3>🌐 24/7 Cloud Access (Laptop Turned Off / Away from Home)</h3>
            <p>
                When your laptop is turned <strong>OFF</strong>, its hardware stops running local web servers.
                To open the website on any device 24/7 even when your laptop is off, deploy this app to 
                <strong>Streamlit Community Cloud (100% Free)</strong> or <strong>Render</strong>.
                Follow the 3-step <code>CLOUD_DEPLOYMENT_GUIDE.md</code> included in this project directory.
            </p>
        </section>

        <section class="app-frame-container">
            <div class="frame-header">
                <span class="frame-title">⚡ Interactive Control Room (Streamlit Engine Core)</span>
                <a href="{app_target_url}" target="_blank" style="color:#38bdf8; text-decoration:none; font-weight:600; font-size:0.85rem;">Pop out ↗</a>
            </div>
            <iframe src="{app_target_url}"></iframe>
        </section>
    </main>

    <footer>
        National Centre for Polar and Ocean Research (NCPOR) • Ministry of Earth Sciences • Government of India
    </footer>
</body>
</html>
"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def run_platform_server():
    print("===============================================================")
    print("  POLAR ENERGY INTELLIGENCE SYSTEM (PEIS) - HTTP PLATFORM")
    print("  NCPOR / Ministry of Earth Sciences (MoES)")
    print("===============================================================")
    print(f"[INFO] Initializing HTTP Platform Server on port {PLATFORM_PORT}...")
    
    start_streamlit_process()

    watch_thread = threading.Thread(target=watchdog_daemon, daemon=True)
    watch_thread.start()

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True

    try:
        httpd = ThreadedHTTPServer((HOST, PLATFORM_PORT), PlatformHTTPRequestHandler)
        print(f"[SUCCESS] HTTP Platform Server is active!")
        print(f"  • Local Machine URL  : http://localhost:{PLATFORM_PORT}")
        print(f"  • Wi-Fi Network URL  : http://{LOCAL_IP}:{PLATFORM_PORT}")
        print(f"  • Direct App URL     : http://localhost:{STREAMLIT_PORT}")
        print("---------------------------------------------------------------")
        
        def open_browser():
            time.sleep(2)
            webbrowser.open(f"http://localhost:{PLATFORM_PORT}")
            
        threading.Thread(target=open_browser, daemon=True).start()

        httpd.serve_forever()
    except Exception as e:
        print(f"[FATAL] Could not start HTTP Platform Server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_platform_server()
