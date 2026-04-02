"""
EnergyML Dashboard Server
- arayuz.html'i sunar
- /api/train endpoint'i ile model egitimini baslatir
- Server-Sent Events ile progress gonderir
"""
import http.server
import gzip
import json
import os
import sys
import threading
import time
import subprocess
import re

PORT = 8080
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Train durumu (thread-safe)
train_state = {
    'running': False,
    'progress': 0,
    'step': '',
    'log': [],
    'result': None,
}
train_lock = threading.Lock()


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.path = '/arayuz.html'
            return super().do_GET()
        elif self.path == '/api/train/status':
            return self.handle_train_status()
        elif self.path == '/api/train/stream':
            return self.handle_train_stream()
        elif self.path.startswith('/output/'):
            if self.path.endswith('.json'):
                return self.serve_gzipped_json()
            return super().do_GET()
        else:
            return super().do_GET()

    def do_POST(self):
        if self.path == '/api/train':
            return self.handle_train_start()
        self.send_error(404)

    def handle_train_start(self):
        with train_lock:
            if train_state['running']:
                self.send_json(400, {'error': 'Egitim zaten devam ediyor'})
                return
            train_state['running'] = True
            train_state['progress'] = 0
            train_state['step'] = 'Baslatiliyor...'
            train_state['log'] = []
            train_state['result'] = None

        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        self.send_json(200, {'status': 'started'})

    def handle_train_status(self):
        with train_lock:
            data = {
                'running': train_state['running'],
                'progress': train_state['progress'],
                'step': train_state['step'],
                'log_count': len(train_state['log']),
                'result': train_state['result'],
            }
        self.send_json(200, data)

    def handle_train_stream(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        last_progress = -1
        while True:
            with train_lock:
                progress = train_state['progress']
                step = train_state['step']
                running = train_state['running']
                result = train_state['result']

            if progress != last_progress:
                data = json.dumps({'progress': progress, 'step': step, 'running': running, 'result': result})
                try:
                    self.wfile.write(f'data: {data}\n\n'.encode())
                    self.wfile.flush()
                except:
                    break
                last_progress = progress

            if not running:
                data = json.dumps({'progress': 100, 'step': 'Tamamlandi!', 'running': False, 'result': result})
                try:
                    self.wfile.write(f'data: {data}\n\n'.encode())
                    self.wfile.flush()
                except:
                    pass
                break

            time.sleep(0.5)

    def serve_gzipped_json(self):
        file_path = os.path.join(BASE_DIR, self.path.lstrip('/'))
        if not os.path.isfile(file_path):
            self.send_error(404)
            return
        try:
            with open(file_path, 'rb') as f:
                raw = f.read()
            accept_encoding = self.headers.get('Accept-Encoding', '')
            if 'gzip' in accept_encoding:
                compressed = gzip.compress(raw)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Encoding', 'gzip')
                self.send_header('Content-Length', str(len(compressed)))
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                self.wfile.write(compressed)
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                self.wfile.write(raw)
        except Exception as e:
            self.send_error(500, str(e))

    def send_json(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # Sessiz


def update_progress(progress, step):
    with train_lock:
        train_state['progress'] = progress
        train_state['step'] = step


def run_training():
    """Pipeline'i calistir ve progress guncelle."""
    try:
        update_progress(5, 'Veri yukleniyor...')

        proc = subprocess.Popen(
            [sys.executable, 'src/main.py'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=BASE_DIR
        )

        step_map = {
            'ADIM 1/7': (10, 'Veri birlestirme ve temizleme...'),
            'ADIM 2/7': (20, 'Feature engineering...'),
            'ADIM 3/7': (30, 'Model egitimi...'),
            'NAIVE BASELINE': (35, 'Naive Baseline...'),
            'RANDOM FOREST': (40, 'Random Forest egitiliyor...'),
            'XGBOOST': (50, 'XGBoost egitiliyor...'),
            'Hiperparametre': (55, 'Hiperparametre optimizasyonu...'),
            'LSTM': (65, 'LSTM egitiliyor...'),
            'ADIM 4/7': (75, 'Akademik dogrulama testleri...'),
            'ADIM 5/7': (85, 'Turkiye tahmini...'),
            'ADIM 6/7': (90, 'Karbon ayak izi...'),
            'ADIM 7/7': (95, 'Grafikler olusturuluyor...'),
            'PIPELINE TAMAMLANDI': (100, 'Tamamlandi!'),
        }

        result_data = {}
        for line in proc.stdout:
            line = line.strip()
            with train_lock:
                train_state['log'].append(line)

            for key, (prog, step) in step_map.items():
                if key in line:
                    update_progress(prog, step)
                    break

            # Sonuclari yakala
            if 'Test MAPE:' in line and 'En iyi model' in line:
                result_data['best_model'] = line
            if 'Turkiye Tahminleri' in line:
                result_data['forecasts'] = []

        proc.wait()

        with train_lock:
            train_state['running'] = False
            train_state['progress'] = 100
            train_state['step'] = 'Tamamlandi!'
            train_state['result'] = {
                'success': proc.returncode == 0,
                'return_code': proc.returncode,
            }

    except Exception as e:
        with train_lock:
            train_state['running'] = False
            train_state['step'] = f'Hata: {str(e)}'
            train_state['result'] = {'success': False, 'error': str(e)}


if __name__ == '__main__':
    print(f'EnergyML Dashboard Server')
    print(f'http://localhost:{PORT}')
    print(f'Durdurmak icin Ctrl+C')

    server = http.server.HTTPServer(('', PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nSunucu durduruldu.')
        server.server_close()
