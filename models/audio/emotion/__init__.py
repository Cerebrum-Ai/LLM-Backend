# This file makes the emotion directory a Python package
# It can also expose specific classes/functions to make imports cleaner

from .audio_processor import SimpleAudioAnalyzer

__all__ = ['SimpleAudioAnalyzer']
