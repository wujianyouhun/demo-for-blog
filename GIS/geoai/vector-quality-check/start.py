import argparse,os,subprocess,sys,time,webbrowser
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--backend-port',type=int,default=8002);p.add_argument('--frontend-port',type=int,default=5175);p.add_argument('--no-install',action='store_true');p.add_argument('--no-browser',action='store_true');a=p.parse_args()
 if not any((ROOT/'data').glob('*.geojson')):subprocess.run([sys.executable,'generate_demo_data.py'],cwd=ROOT,check=True)
 if not a.no_install and not (ROOT/'frontend/node_modules').exists():subprocess.run(['npm','install'],cwd=ROOT/'frontend',check=True,shell=os.name=='nt')
 b=subprocess.Popen([sys.executable,'-m','uvicorn','backend.main:app','--host',a.host,'--port',str(a.backend_port)],cwd=ROOT);f=subprocess.Popen(['npm','run','dev','--','--host',a.host,'--port',str(a.frontend_port)],cwd=ROOT/'frontend',shell=os.name=='nt')
 if not a.no_browser:time.sleep(2);webbrowser.open(f'http://{a.host}:{a.frontend_port}')
 try:return b.wait()
 except KeyboardInterrupt:b.terminate();f.terminate();return 0
if __name__=='__main__':raise SystemExit(main())
