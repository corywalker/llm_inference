import re
import math
import matplotlib.pyplot as plt
import sys
import os

def parse_file(filepath):
    """
    Parses a file to extract tensor information.
    Returns a dict mapping tensor_name -> {'sum': float, 'values': list[float]}
    or a list of such dictionaries if order matters. 
    Here we return a list to preserve order for plotting, 
    but we also want to lookup by name.
    
    Let's return a list of dicts: {'name': str, 'sum': float, 'values': list[float], 'line_num': int}
    """
    tensors = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    current_tensor = None
    capture_values = False
    
    # Regex patterns
    # Matches "sum = 28.737751" or scientific notation
    # Ensure there is at least one digit
    sum_pattern = re.compile(r'^\s+sum\s+=\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)')
    
    # Matches "attn_norm-0 = ..." or "attn_norm-0 = {..."
    # We assume the name is at the start of the line (after potential whitespace) up to " ="
    # In the example: "attn_norm-0 = ..."
    name_pattern = re.compile(r'^\s*([a-zA-Z0-9_\.-]+)\s+=')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for sum
        sum_match = sum_pattern.match(line)
        if sum_match:
            if current_tensor:
                # We found a new sum, but haven't finished the previous tensor? 
                # Actually usually formats are block based.
                # If we encounter a sum, it's likely the start of a new tensor block description
                # or the end of one.
                # In the grep output: "sum = ..." comes BEFORE "attn_norm-0 = ..."
                pass

            tensor_sum = float(sum_match.group(1))
            
            # The next line should have the name
            if i + 1 < len(lines):
                name_line = lines[i+1]
                name_match = name_pattern.match(name_line)
                if name_match:
                    name = name_match.group(1)
                    current_tensor = {
                        'name': name,
                        'sum': tensor_sum,
                        'values': [],
                        'line_num': i + 1
                    }
                    tensors.append(current_tensor)
                    i += 1 # Advance to name line
                    capture_values = True
                else:
                    # Maybe name is on the same line? Unlikely based on grep.
                    # Or maybe skipping some lines.
                    pass
        
        elif capture_values and current_tensor:
            # We are inside a tensor block, looking for values like "[ 0.123, ... ]"
            # format: [      0.1653,      -0.2382, ...
            
            # Check if block ends
            # If we see a new "sum =" or "name =", we stop? 
            # Or matches "]" at start of line?
            # The grep shows nested brackets [[ [ ... ] ]].
            
            # Simplest approach: extract all floats from the line
            # If line has no floats, maybe it's just formatting. 
            # Stop if we hit a line that looks like a new header or is empty for too long?
            # Actually, let's just grab valid floats until we hit new metadata.
            
            if sum_pattern.match(line) or name_pattern.match(line):
                # Oops, we ran into the next block without closing the previous one?
                # This should be handled by the main loop anyway, but we need to reset flag.
                capture_values = False
                # Don't increment i here, let the loop re-process this line as a start
                continue
                
            vals = re.findall(r'([-+]?\d*\.\d+|[-+]?\d+)', line)
            # Filter out timestamps or line numbers if any (unlikely in this format)
            # Only take things that look like tensor values.
            # The regex finds floats.
            
            valid_vals = []
            for v in vals:
                try:
                    f = float(v)
                    # Simple heuristic: ignore integers that look like dims if they appear?
                    # But values can be anything.
                    valid_vals.append(f)
                except:
                    pass
            
            if valid_vals:
                current_tensor['values'].extend(valid_vals)
                
        i += 1
        
    return tensors

def calculate_metrics(ref_tensor, test_tensor):
    sum_diff = abs(ref_tensor['sum'] - test_tensor['sum'])
    
    # Compare values
    ref_vals = ref_tensor['values']
    test_vals = test_tensor['values']
    
    n = min(len(ref_vals), len(test_vals))
    if n == 0:
        return sum_diff, 0.0
        
    # MSE
    mse = 0.0
    for j in range(n):
        d = ref_vals[j] - test_vals[j]
        mse += d*d
    mse /= n
    
    return sum_diff, mse

def main():
    file1 = 'tmp/llm_inference_out.txt'
    file2 = 'tmp/llama_cpp_out.txt'  # Reference
    
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"Error: One or both files not found: {file1}, {file2}")
        return

    print(f"Parsing {file1}...")
    tensors1 = parse_file(file1)
    print(f"Found {len(tensors1)} tensors in {file1}")
    
    print(f"Parsing {file2}...")
    tensors2 = parse_file(file2)
    print(f"Found {len(tensors2)} tensors in {file2}")
    
    # Create lookup for tensors2 (Reference)
    # Since names might repeat (e.g. layers), we shouldn't just use a dict.
    # But presumably the sequence is the same.
    # Let's try to align by name first, if unique. If duplicates, use order.
    
    # Actually, "attn_norm-0" implies sequence.
    # Let's assume strict sequence matching for names.
    
    # Map name -> list of tensors
    tensors2_map = {}
    for t in tensors2:
        if t['name'] not in tensors2_map:
            tensors2_map[t['name']] = []
        tensors2_map[t['name']].append(t)
        
    # Keep track of index used for each name
    tensors2_idx = {name: 0 for name in tensors2_map}
    
    results = []
    
    print("Comparing tensors...")
    
    for t1 in tensors1:
        name = t1['name']
        if name in tensors2_map:
            idx = tensors2_idx[name]
            if idx < len(tensors2_map[name]):
                t2 = tensors2_map[name][idx]
                tensors2_idx[name] += 1
                
                sum_diff, weight_mse = calculate_metrics(t2, t1) # t2 is ref
                
                results.append({
                    'name': name,
                    'sum_diff': sum_diff,
                    'weight_mse': weight_mse,
                    'idx': len(results)
                })
            else:
                print(f"Warning: More instances of {name} in file1 than file2")
        else:
            print(f"Warning: {name} not found in reference file")

    if not results:
        print("No matching tensors found to compare.")
        return

    # Plotting
    indices = [r['idx'] for r in results]
    sum_diffs = [r['sum_diff'] for r in results]
    weight_mses = [r['weight_mse'] for r in results]
    names = [r['name'] for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Tensor Index')
    ax1.set_ylabel('Sum Deviation (Abs Diff)', color=color)
    ax1.plot(indices, sum_diffs, color=color, label='Sum Diff', marker='o', markersize=2, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log') # Log scale often helps for discrepancies

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Weight MSE', color=color)  # we already handled the x-label with ax1
    ax2.plot(indices, weight_mses, color=color, label='Weight MSE', marker='x', markersize=2, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')

    plt.title('Tensor Deviations: llm_inference vs llama.cpp (Reference)')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    out_png = 'tmp/tensor_comparison.png'
    plt.savefig(out_png)
    print(f"Plot saved to {out_png}")

    # Print top top deviations
    print("\nTop 5 Sum Deviations:")
    sorted_by_sum = sorted(results, key=lambda x: x['sum_diff'], reverse=True)
    for r in sorted_by_sum[:5]:
        print(f"  {r['name']} (idx {r['idx']}): {r['sum_diff']}")

    print("\nTop 5 Weight MSEs:")
    sorted_by_mse = sorted(results, key=lambda x: x['weight_mse'], reverse=True)
    for r in sorted_by_mse[:5]:
        print(f"  {r['name']} (idx {r['idx']}): {r['weight_mse']}")

if __name__ == "__main__":
    main()
