import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile
# AI models / ChatGPT sign-in data of every test server go to a throwaway folder, never the user's Application Support
os.environ.setdefault("BODYGLASS_HOME", tempfile.mkdtemp(prefix="bg_test_home_"))
