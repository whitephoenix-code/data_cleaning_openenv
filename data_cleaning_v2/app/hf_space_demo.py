"""
app/hf_space_demo.py — Hugging Face Spaces demo entry point.
Runs the same FastAPI server, optimized for HF Spaces environment.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app  # re-export the FastAPI app
