#!/usr/bin/env python3

from src.pepino.analysis.service import analysis_service

def test_discord_template():
    with analysis_service() as service:
        result = service.pulsecheck(output_format='discord')
        print(f"Length: {len(result)} characters")
        print("=" * 50)
        print("Preview:")
        print(result[:1000] + "..." if len(result) > 1000 else result)
        print("=" * 50)
        print(f"Fits in Discord: {'✅' if len(result) <= 2000 else '❌'}")

if __name__ == "__main__":
    test_discord_template() 