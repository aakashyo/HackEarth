import requests
import sys
import time
import os

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

API_URL = "http://localhost:8000/predict"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD} 🧠  Multilingual Emotion AI - Terminal Interface{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f" {Colors.YELLOW}• Connecting to Brain at: {API_URL}{Colors.ENDC}")
    print(f" {Colors.YELLOW}• Supports: Tamil, Hindi, Bengali, English{Colors.ENDC}")
    print(f" {Colors.YELLOW}• Type 'exit' to quit{Colors.ENDC}\n")

    # Check connection first
    try:
        requests.get("http://localhost:8000/health", timeout=2)
        print(f"{Colors.GREEN} ✓ Connected to Neural Backend{Colors.ENDC}\n")
    except:
        print(f"{Colors.RED} ❌ Error: Backend is not running!{Colors.ENDC}")
        print(f"    Please run 'python app.py' in the 'backend' folder first.")
        return

    while True:
        try:
            text = input(f"{Colors.CYAN}📝 Enter text > {Colors.ENDC}").strip()
            
            if text.lower() in ['exit', 'quit']:
                print(f"\n{Colors.HEADER}Goodbye! 👋{Colors.ENDC}")
                break
            
            if not text:
                continue

            print(f"   {Colors.BLUE}Thinking...{Colors.ENDC}")
            
            try:
                start_time = time.time()
                response = requests.post(API_URL, json={"text": text}, timeout=10)
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    
                    p_emo = data.get('primary_emotion', 'N/A')
                    s_emo = data.get('secondary_emotion', 'N/A')
                    p_conf = float(data.get('primary_confidence', 0.0)) * 100
                    s_conf = float(data.get('secondary_confidence', 0.0)) * 100
                    
                    print(f"\n   {Colors.BOLD}✨ PREDICTION RESULTS ({int(latency)}ms):{Colors.ENDC}")
                    print(f"   {'-'*40}")
                    
                    # Primary
                    print(f"   🎯 {Colors.BOLD}Primary Tone:{Colors.ENDC}   {Colors.GREEN}{p_emo:<15}{Colors.ENDC} [{p_conf:.1f}%]")
                    
                    # Secondary
                    print(f"   🌊 {Colors.BOLD}Underlying Mood:{Colors.ENDC} {Colors.CYAN}{s_emo:<15}{Colors.ENDC} [{s_conf:.1f}%]")
                    print(f"   {'-'*40}\n")
                    
                else:
                    print(f"   {Colors.RED}❌ Server Error: {response.status_code}{Colors.ENDC}")
            except requests.exceptions.ConnectionError:
                 print(f"   {Colors.RED}❌ Error: Lost connection to backend.{Colors.ENDC}")
                 
        except KeyboardInterrupt:
            print(f"\n\n{Colors.HEADER}Goodbye! 👋{Colors.ENDC}")
            break

if __name__ == "__main__":
    main()
