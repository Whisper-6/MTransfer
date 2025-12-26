import torch

def test_contiguous_error():
    """
    Minimal reproduction script to simulate the (*bias) last dimension must be contiguous error.
    This mimics what happens inside torch.nn.functional.scaled_dot_product_attention (SDPA)
    when passed a non-contiguous mask.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    B, H, Q, K = 2, 4, 32, 32
    
    # 1. Simulate query/key/value
    query = torch.randn(B, H, Q, 64, device=device, dtype=dtype)
    key = torch.randn(B, H, K, 64, device=device, dtype=dtype)
    value = torch.randn(B, H, K, 64, device=device, dtype=dtype)
    
    # 2. Simulate attention mask behavior from solve_bias.py
    # Scenario A: Contiguous mask (Should pass)
    mask_contiguous = torch.zeros(B, 1, Q, K, device=device, dtype=dtype)
    mask_contiguous = mask_contiguous.expand(-1, H, -1, -1).contiguous()
    
    print("Testing contiguous mask...")
    try:
        torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask_contiguous
        )
        print("PASS: Contiguous mask works.")
    except Exception as e:
        print(f"FAIL: Contiguous mask failed: {e}")

    # Scenario B: Non-contiguous mask (Should fail with specific error)
    # Creating a non-contiguous tensor by slicing/expanding without contiguous()
    mask_base = torch.zeros(B, 1, Q, K, device=device, dtype=dtype)
    mask_expanded = mask_base.expand(-1, H, -1, -1) # Non-contiguous view
    
    print("\nTesting non-contiguous mask (simulating current error)...")
    try:
        torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask_expanded
        )
        print("PASS: Non-contiguous mask works (unexpected for SDPA backend depending on version).")
    except Exception as e:
        print(f"FAIL: Non-contiguous mask failed as expected: {e}")
        
    # Scenario C: Mixed dtype or weird strides
    # Sometimes 'bias' error refers to the mask being implicitly treated as bias
    
if __name__ == "__main__":
    test_contiguous_error()

