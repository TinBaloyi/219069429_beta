"""
Quick setup script to enable AI recommendations
Run: python setup_ai_recommendations.py
"""
import os
import sys
from pathlib import Path

def check_api_keys():
    """Check if API keys are set"""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    provider = os.getenv("AI_PROVIDER", "none").lower()
    
    print("\n" + "="*60)
    print("AI Provider Configuration")
    print("="*60)
    
    if provider == "openai" and openai_key:
        print("✓ OpenAI configured")
        print(f"  Key: {openai_key[:10]}...{openai_key[-4:]}")
        return True
    elif provider == "anthropic" and anthropic_key:
        print("✓ Anthropic configured")
        print(f"  Key: {anthropic_key[:10]}...{anthropic_key[-4:]}")
        return True
    elif provider == "ollama":
        print("✓ Ollama configured (checking connection...)")
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                models = r.json().get("models", [])
                print(f"  Available models: {[m['name'] for m in models]}")
                return True
            else:
                print("  ✗ Ollama not responding")
                return False
        except Exception as e:
            print(f"  ✗ Ollama not available: {e}")
            return False
    else:
        print("⚠ No AI provider configured")
        print(f"  Current provider: {provider}")
        print("\nTo enable AI recommendations, set environment variables:")
        print("\n  Option 1 - OpenAI:")
        print("    set OPENAI_API_KEY=sk-your-key")
        print("    set AI_PROVIDER=openai")
        print("\n  Option 2 - Anthropic:")
        print("    set ANTHROPIC_API_KEY=sk-your-key")
        print("    set AI_PROVIDER=anthropic")
        print("\n  Option 3 - Ollama (Free, Local):")
        print("    Install from https://ollama.ai")
        print("    ollama pull llama2")
        print("    set AI_PROVIDER=ollama")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)
    
    provider = os.getenv("AI_PROVIDER", "none").lower()
    
    # Core required packages
    core = {
        "requests": "Core HTTP library"
    }
    
    # Provider-specific packages
    provider_packages = {
        "openai": {"openai": "OpenAI provider"},
        "anthropic": {"anthropic": "Anthropic provider"},
        "ollama": {}  # Ollama doesn't need extra packages beyond requests
    }
    
    # Check core packages first
    missing = []
    for package, purpose in core.items():
        try:
            __import__(package)
            print(f"✓ {package:12} - {purpose}")
        except ImportError:
            print(f"✗ {package:12} - {purpose} (REQUIRED)")
            missing.append(package)
    
    # Check provider-specific packages
    if provider in provider_packages:
        for package, purpose in provider_packages[provider].items():
            try:
                __import__(package)
                print(f"✓ {package:12} - {purpose}")
            except ImportError:
                print(f"✗ {package:12} - {purpose} (REQUIRED for {provider})")
                missing.append(package)
    
    # Check optional packages
    optional = {
        "openai": "OpenAI provider (optional)",
        "anthropic": "Anthropic provider (optional)"
    }
    
    print("\nOptional packages:")
    for package, purpose in optional.items():
        if package not in provider_packages.get(provider, {}):
            try:
                __import__(package)
                print(f"✓ {package:12} - {purpose}")
            except ImportError:
                print(f"○ {package:12} - {purpose} (not needed for current provider)")
    
    if missing:
        print("\n⚠ To install missing required packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All required dependencies installed!")
    return True

def verify_file_structure():
    """Verify AI recommendations file is in place"""
    print("\n" + "="*60)
    print("Checking File Structure")
    print("="*60)
    
    ai_rec_file = Path("pipelines/ai_recommendations.py")
    if not ai_rec_file.exists():
        ai_rec_file = Path("ai_recommendations.py")
    
    if ai_rec_file.exists():
        print(f"✓ AI recommendations module found: {ai_rec_file}")
    else:
        print("✗ ai_recommendations.py not found!")
        print("  Place the file in project root or pipelines/ folder")
        return False
    
    main_file = Path("pipelines/main.py")
    if main_file.exists():
        content = main_file.read_text()
        if "generate_recommendations_for_event" in content:
            print("✓ main.py already imports AI recommendations")
        else:
            print("⚠ main.py needs to import AI recommendations")
            print("\n  Add this to the imports section:")
            print("  " + "-"*56)
            print("""  try:
      from ai_recommendations import generate_recommendations_for_event
      AI_RECOMMENDATIONS_AVAILABLE = True
  except ImportError:
      AI_RECOMMENDATIONS_AVAILABLE = False
      def generate_recommendations_for_event(event, **kwargs):
          return event.get("recommendations", [])""")
            print("  " + "-"*56)
    else:
        print("✗ pipelines/main.py not found")
        return False
    
    return True

def test_ai_generation():
    """Test AI recommendation generation"""
    print("\n" + "="*60)
    print("Testing AI Generation")
    print("="*60)
    
    try:
        # Import the module
        sys.path.insert(0, str(Path.cwd()))
        sys.path.insert(0, str(Path.cwd() / "pipelines"))
        
        from ai_recommendations import get_ai_engine
        
        # Test bottleneck
        test_bottleneck = {
            "bottleneck_type": "High Error Rate",
            "entity": "Machine_A",
            "severity_score": 75.5,
            "metric": "Error_Rate_%",
            "metric_value": 8.5,
            "reason": "Error rate significantly above threshold"
        }
        
        print("\nTest bottleneck:")
        print(f"  Type: {test_bottleneck['bottleneck_type']}")
        print(f"  Entity: {test_bottleneck['entity']}")
        print(f"  Severity: {test_bottleneck['severity_score']}")
        
        # Generate recommendations
        engine = get_ai_engine()
        recs = engine.generate_recommendations(test_bottleneck, max_recommendations=3)
        
        print(f"\n✓ Generated {len(recs)} recommendations:")
        for i, rec in enumerate(recs, 1):
            print(f"\n  {i}. [{rec['priority']}] {rec['action']}")
            print(f"     Verification: {rec['verification']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_env_file():
    """Create a .env template"""
    env_file = Path(".env.example")
    
    content = """# AI Configuration
AI_PROVIDER=openai          # Options: openai, anthropic, ollama, none
OPENAI_API_KEY=sk-xxx       # Get from https://platform.openai.com/api-keys
ANTHROPIC_API_KEY=sk-xxx    # Get from https://console.anthropic.com/

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bottleneck_db
DB_USER=postgres
DB_PASSWORD=your_password

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
"""
    
    env_file.write_text(content)
    print(f"\n✓ Created {env_file}")
    print("  Copy to .env and fill in your values:")
    print("  copy .env.example .env")

def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("AI Recommendations Setup Check")
    print("="*60)
    
    checks = []
    
    # Run checks
    checks.append(("API Keys", check_api_keys()))
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("File Structure", verify_file_structure()))
    
    # Create env template
    if not Path(".env.example").exists():
        create_env_file()
    
    # Test if everything is ready
    if all(result for _, result in checks):
        print("\n" + "="*60)
        print("Running Integration Test")
        print("="*60)
        checks.append(("AI Generation", test_ai_generation()))
    
    # Summary
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    
    for name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} - {name}")
    
    all_passed = all(result for _, result in checks)
    
    if all_passed:
        print("\n🎉 All checks passed! AI recommendations are ready to use.")
        print("\nNext steps:")
        print("  1. Start the server: python -m uvicorn pipelines.main:app --reload")
        print("  2. Run analysis: curl http://localhost:8000/analysis/logistics")
        print("  3. Check recommendations in the response JSON")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("\nQuick fixes:")
        print("  1. Install packages: pip install openai anthropic requests")
        print("  2. Set API key: set OPENAI_API_KEY=sk-xxx")
        print("  3. Set provider: set AI_PROVIDER=openai")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()