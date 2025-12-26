"""
Main script to run complete TNM analysis pipeline
1. Scrape Wikipedia data
2. Generate TNM broadband dataset
3. Run full analysis
"""
import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script and handle errors"""
    try:
        print(f"\n{'='*60}")
        print(f"Running: {script_name}")
        print(f"{'='*60}\n")
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True,
                              check=True)
        print(f"\n[OK] {script_name} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_name}: {e}\n")
        return False
    except FileNotFoundError:
        print(f"\n✗ File not found: {script_name}\n")
        return False

def main():
    """Main execution function"""
    print("=" * 60)
    print("TNM BROADBAND ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Step 1: Scrape Wikipedia (optional - may fail if no internet)
    print("\n[Optional] Scraping TNM information from Wikipedia...")
    run_script("scrape_tnm_wikipedia.py")
    
    # Step 2: Generate TNM broadband dataset
    print("\n[Required] Generating TNM broadband customer dataset...")
    if not run_script("generate_tnm_data.py"):
        print("ERROR: Could not generate dataset. Exiting.")
        return
    
    # Check if dataset was created
    if not os.path.exists("tnm_broadband.csv"):
        print("ERROR: Dataset file not found. Exiting.")
        return
    
    # Step 3: Run analysis
    print("\n[Required] Running TNM broadband data analysis...")
    run_script("tnm_analysis.py")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - tnm_broadband.csv (dataset)")
    print("  - tnm_revenue_by_plan.png")
    print("  - tnm_data_usage_vs_churn.png")
    print("  - tnm_downtime_vs_churn.png")
    print("  - tnm_churn_by_region.png")
    print("  - tnm_churn_by_plan.png")
    if os.path.exists("tnm_wikipedia_data.json"):
        print("  - tnm_wikipedia_data.json (Wikipedia data)")

if __name__ == "__main__":
    main()

