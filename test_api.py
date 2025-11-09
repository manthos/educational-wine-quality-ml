"""
Test script for Wine Quality Prediction API
"""
import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

# Test data - example wine
test_wine = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "ph": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

# High quality wine example
high_quality_wine = {
    "fixed_acidity": 7.8,
    "volatile_acidity": 0.28,
    "citric_acid": 0.56,
    "residual_sugar": 2.3,
    "chlorides": 0.075,
    "free_sulfur_dioxide": 17.0,
    "total_sulfur_dioxide": 60.0,
    "density": 0.998,
    "ph": 3.16,
    "sulphates": 0.68,
    "alcohol": 11.5
}

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*60)
    print("Testing /predict endpoint (Single prediction)")
    print("="*60)
    print(f"Input wine features:")
    print(json.dumps(test_wine, indent=2))
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_wine
    )
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_high_quality_prediction():
    """Test prediction with high quality wine"""
    print("\n" + "="*60)
    print("Testing /predict with high quality wine")
    print("="*60)
    print(f"Input wine features:")
    print(json.dumps(high_quality_wine, indent=2))
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=high_quality_wine
    )
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing /predict/batch endpoint")
    print("="*60)
    batch_wines = [test_wine, high_quality_wine]
    print(f"Number of wines: {len(batch_wines)}")
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=batch_wines
    )
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_invalid_input():
    """Test with invalid input"""
    print("\n" + "="*60)
    print("Testing /predict with invalid input")
    print("="*60)
    invalid_wine = {
        "fixed_acidity": -1,  # Invalid: negative value
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "ph": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    print(f"Input with invalid value (fixed_acidity = -1):")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=invalid_wine
    )
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    print("Expected: Validation error (422)")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Wine Quality Prediction API - Test Suite")
    print("="*60)
    print("\nMake sure the API is running at http://localhost:8000")
    print("Run: python predict.py")
    print("="*60)
    
    try:
        # Run tests
        tests = [
            ("Health Check", test_health),
            ("Single Prediction", test_single_prediction),
            ("High Quality Wine", test_high_quality_prediction),
            ("Batch Prediction", test_batch_prediction),
            ("Invalid Input", test_invalid_input),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, "✓ PASSED" if success else "✗ FAILED"))
            except Exception as e:
                results.append((test_name, f"✗ ERROR: {str(e)}"))
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for test_name, result in results:
            print(f"{test_name:.<40} {result}")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the server is running: python predict.py")
