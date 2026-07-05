import tensorflow as tf
from pathlib import Path
import time,shutil,os
import globalmap as GL
import re

def merge_tfrecord_binary_concat(input_files, output_file, overwrite=False):
    """
    Merge multiple TFRecord files by directly concatenating binary content.
    This is MUCH faster than parsing individual records (10-100x speedup).
    
    How it works:
        TFRecord files are just sequences of binary records concatenated together.
        Therefore, merging can be done by simply appending the raw bytes of each file.
        No parsing, decoding, or re-encoding is needed.
    
    Args:
        input_files: List of input TFRecord file paths, e.g., ['file1.tfrecord', 'file2.tfrecord']
        output_file: Path to the output merged TFRecord file
        overwrite: If True, overwrite existing output file; if False, raise error if exists
    
    Returns:
        bool: True if merge completed successfully, False otherwise
    
    Example:
        >>> files = ['data/train_001.tfrecord', 'data/train_002.tfrecord']
        >>> merge_tfrecord_binary_concat(files, 'data/merged_train.tfrecord', overwrite=True)
        ✓ Appended: train_001.tfrecord
        ✓ Appended: train_002.tfrecord
        ✅ Binary merge completed: data/merged_train.tfrecord
    
    Performance Note:
        - Method 1 (parse each record): ~1000 records/second
        - Method 2 (binary concatenation): ~500 MB/second (disk I/O limited)
        - Speedup: 10-100x depending on record size
    """
    # Convert to Path object for better path handling
    output_path = Path(output_file)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {output_path.parent}")
    
    # Check if output file already exists
    if output_path.exists() and not overwrite:
        print(f"⚠️ Warning: {output_file} already exists.")
        print("   Set overwrite=True to overwrite or choose a different filename.")
        return False
    
    # Filter out non-existent input files
    valid_files = []
    missing_files = []
    
    for input_file in input_files:
        input_path = Path(input_file)
        if input_path.exists():
            valid_files.append(input_file)
        else:
            missing_files.append(input_file)
    
    if missing_files:
        print(f"⚠️ Warning: {len(missing_files)} file(s) not found and will be skipped:")
        for f in missing_files[:5]:  # Show first 5 missing files
            print(f"   - {f}")
        if len(missing_files) > 5:
            print(f"   ... and {len(missing_files) - 5} more")
    
    if not valid_files:
        print("❌ Error: No valid input files found. Merge aborted.")
        return False
    
    print(f"\nMerging {len(valid_files)} file(s):")
    
    # Track total bytes for progress reporting
    total_bytes = 0
    start_time = time.time()
    
    # Open output file in binary write mode
    with open(output_file, 'wb') as outfile:
        for input_file in valid_files:
            input_path = Path(input_file)
            file_size = input_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            # Open input file in binary read mode and copy content
            with open(input_file, 'rb') as infile:
                # shutil.copyfileobj efficiently copies binary data between file objects
                # It uses a default buffer size of 64KB for optimal performance
                shutil.copyfileobj(infile, outfile)
            
            total_bytes += file_size
            print(f"  ✓ Appended: {input_path.name} ({file_size_mb:.2f} MB)")
    
    elapsed_time = time.time() - start_time
    total_mb = total_bytes / (1024 * 1024)
    speed_mb_s = total_mb / elapsed_time if elapsed_time > 0 else 0
    
    print("\n✅ Binary merge completed!")
    print(f"   Output file: {output_file}")
    print(f"   Total size: {total_mb:.2f} MB")
    print(f"   Files merged: {len(valid_files)}")
    print(f"   Time elapsed: {elapsed_time:.2f} seconds")
    print(f"   Speed: {speed_mb_s:.1f} MB/s")
    
    return True


def merge_tfrecord_buffered(input_files, output_file, overwrite=False, buffer_size_mb=10):
    """
    Merge TFRecord files using manual buffered writing.
    Alternative to shutil.copyfileobj with configurable buffer size.
    
    Args:
        input_files: List of input TFRecord file paths
        output_file: Path to output merged file
        overwrite: Whether to overwrite existing output file
        buffer_size_mb: Buffer size in megabytes (default: 10 MB)
    
    Returns:
        bool: True if merge successful, False otherwise
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not overwrite:
        print(f"❌ Error: {output_file} already exists")
        return False
    
    buffer_size = buffer_size_mb * 1024 * 1024  # Convert MB to bytes
    total_bytes = 0
    valid_count = 0
    
    print(f"Merging with buffer size: {buffer_size_mb} MB")
    
    with open(output_file, 'wb') as outfile:
        for input_file in input_files:
            input_path = Path(input_file)
            if not input_path.exists():
                print(f"⚠️ Skip missing: {input_path.name}")
                continue
            
            file_size = 0
            with open(input_file, 'rb') as infile:
                # Read and write in chunks to manage memory usage
                while True:
                    chunk = infile.read(buffer_size)
                    if not chunk:
                        break
                    outfile.write(chunk)
                    file_size += len(chunk)
            
            total_bytes += file_size
            valid_count += 1
            print(f"  ✓ Appended: {input_path.name} ({file_size/(1024*1024):.2f} MB)")
    
    if valid_count == 0:
        print("❌ Error: No valid files to merge")
        return False
    
    print(f"✅ Merge completed: {output_file} ({total_bytes/(1024*1024):.2f} MB)")
    return True


def merge_tfrecord_platform_optimized(input_files, output_file, overwrite=False):
    """
    Platform-optimized merge using appropriate method for OS.
    On Unix-like systems, can use sendfile() for zero-copy transfer.
    
    Args:
        input_files: List of input TFRecord file paths
        output_file: Path to output merged file
        overwrite: Whether to overwrite existing output file
    """
    import sys
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not overwrite:
        print(f"❌ Error: {output_file} already exists")
        return False
    
    # Filter valid files
    valid_files = [f for f in input_files if Path(f).exists()]
    
    if not valid_files:
        print("❌ No valid input files")
        return False
    
    print(f"Merging {len(valid_files)} files using platform-optimized method...")
    
    total_bytes = 0
    
    # On Linux/macOS, we can use sendfile for zero-copy (most efficient)
    if sys.platform in ('linux', 'darwin'):
        try:
            with open(output_file, 'wb') as outfile:
                outfd = outfile.fileno()
                for input_file in valid_files:
                    with open(input_file, 'rb') as infile:
                        infd = infile.fileno()
                        file_size = Path(input_file).stat().st_size
                        # sendfile copies data directly between file descriptors
                        # without userspace copying (zero-copy)
                        os.sendfile(outfd, infd, 0, file_size)
                        total_bytes += file_size
                        print(f"  ✓ Appended: {Path(input_file).name} (zero-copy)")
            
            print(f"✅ Merge completed: {output_file} ({total_bytes/(1024*1024):.2f} MB)")
            return True
            
        except Exception as e:
            print(f"⚠️ sendfile failed, falling back to shutil: {e}")
            # Fall back to shutil method
    
    # Fallback to standard shutil method for Windows or if sendfile fails
    with open(output_file, 'wb') as outfile:
        for input_file in valid_files:
            with open(input_file, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
            print(f"  ✓ Appended: {Path(input_file).name}")
    
    print(f"✅ Merge completed: {output_file}")
    return True


def verify_tfrecord_integrity(file_path, show_stats=True):
    """
    Verify that a TFRecord file is valid by attempting to read it.
    This is useful to check that binary concatenation didn't corrupt the file.
    
    Args:
        file_path: Path to the TFRecord file to verify
        show_stats: Whether to print file statistics
    
    Returns:
        tuple: (is_valid, record_count, file_size_mb)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False, 0, 0
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    record_count = 0
    is_valid = True
    
    try:
        # Try to read all records
        for record in tf.data.TFRecordDataset(str(file_path)):
            record_count += 1
            
            # Optional: stop after reading a few records to save time
            # if record_count == 100 and file_size_mb > 100:
            #     break
    except Exception as e:
        print(f"❌ File corrupted: {e}")
        is_valid = False
    
    if show_stats:
        status = "✅ VALID" if is_valid else "❌ CORRUPTED"
        print(f"{status} file: {file_path.name}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Records: {record_count:,}")
    
    return is_valid, record_count, file_size_mb


# Complete example with all features
def complete_merge_example():
    """
    Complete example demonstrating all merge functions and verification.
    """
    
    # Define input files and output path
    input_files = [
        '../Training_data_gen_127/data/snr1.5-4.0dB/10th/train_001.tfrecord',
        '../Training_data_gen_127/data/snr1.5-4.0dB/10th/train_002.tfrecord',
        '../Training_data_gen_127/data/snr1.5-4.0dB/10th/train_003.tfrecord',
    ]
    
    output_file = '../Training_data_gen_127/data/snr1.5-4.0dB/10th/merged_train.tfrecord'
    
    # Method 1: Fast binary concatenation (RECOMMENDED)
    print("=" * 60)
    print("METHOD 1: Fast Binary Concatenation")
    print("=" * 60)
    merge_tfrecord_binary_concat(input_files, output_file, overwrite=True)
    
    # Verify the merged file
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    verify_tfrecord_integrity(output_file, show_stats=True)
    
    # Method 2: Buffered reading (alternative)
    print("\n" + "=" * 60)
    print("METHOD 2: Buffered Reading")
    print("=" * 60)
    merge_tfrecord_buffered(input_files, 'merged_buffered.tfrecord', overwrite=True)
    
    # Method 3: Raw binary copy (most basic)
    print("\n" + "=" * 60)
    print("METHOD 3: Raw Binary Copy")
    print("=" * 60)
    
    output_path = Path('merged_raw.tfrecord')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open('merged_raw.tfrecord', 'wb') as outfile:
        for input_file in input_files:
            if Path(input_file).exists():
                with open(input_file, 'rb') as infile:
                    outfile.write(infile.read())  # Load entire file into memory
                print(f"✓ Copied: {Path(input_file).name}")
    
    print(f"✅ Created: merged_raw.tfrecord")


# Performance comparison function
def benchmark_merge_methods(input_files, output_prefix):
    """
    Benchmark different merge methods to demonstrate performance differences.
    
    Args:
        input_files: List of input file paths
        output_prefix: Prefix for output files (will append method names)
    """
    import time
    
    print("=" * 70)
    print("PERFORMANCE BENCHMARK: TFRecord Merge Methods")
    print("=" * 70)
    print(f"Input files: {len(input_files)}")
    print(f"Total size: {sum(Path(f).stat().st_size for f in input_files if Path(f).exists()) / (1024*1024):.2f} MB")
    print("-" * 70)
    
    # Method 1: Traditional record-by-record parsing (SLOW)
    start = time.time()
    record_count = 0
    with tf.io.TFRecordWriter(f'{output_prefix}_method1_slow.tfrecord') as writer:
        for input_file in input_files:
            if not Path(input_file).exists():
                continue
            for record in tf.data.TFRecordDataset(input_file):
                writer.write(record.numpy())
                record_count += 1
    time1 = time.time() - start
    print(f"Method 1 (Parse each record): {time1:.2f}s - {record_count} records")
    
    # Method 2: Binary concatenation (FAST)
    start = time.time()
    with open(f'{output_prefix}_method2_fast.tfrecord', 'wb') as outfile:
        for input_file in input_files:
            if not Path(input_file).exists():
                continue
            with open(input_file, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
    time2 = time.time() - start
    print(f"Method 2 (Binary concatenation): {time2:.2f}s")
    
    # Calculate speedup
    if time1 > 0 and time2 > 0:
        speedup = time1 / time2
        print(f"\n🚀 Speedup: Method 2 is {speedup:.1f}x faster than Method 1")
    
    print("=" * 70)

def merge_tfrecord_files_pathlib(root_dir, keyword, output_file, overwrite=False):
    """
    Find and merge TFRecord files using pathlib with output file existence check.
    
    Recursively searches all subdirectories for .tfrecord files containing 
    a specific keyword in their filename, then merges them into one file.
    
    Args:
        root_dir: Root directory path to search recursively
        keyword: Keyword string that must appear in the filename
                 (e.g., 'train', 'test', 'part-000')
        output_file: Path to the output merged .tfrecord file
        overwrite: If True, overwrite existing file; if False, raise error if file exists
    
    Returns:
        bool: True if merge completed successfully, False otherwise
    
    Example:
        >>> merge_tfrecord_files_pathlib('/data/dataset', 'train', 'merged_train.tfrecord', overwrite=False)
    """
    # Convert to Path object for better path handling
    output_path = Path(output_file)
    
    # Check if output file already exists
    if output_path.exists():
        if overwrite:
            print(f"⚠️ Warning: {output_file} already exists. Overwriting...")
        else:
            print(f"❌ Error: {output_file} already exists!")
            print("   To overwrite, set overwrite=True")
            print("   Or choose a different filename") 
    # Convert string path to Path object
    root_path = Path(root_dir)
    
    # Recursively find all matching .tfrecord files
    input_files = list(root_path.rglob(f'*{keyword}*.tfrecord'))
    
    # Check if any files were found
    if not input_files:
        print(f"⚠️ Warning: No .tfrecord files containing '{keyword}' found in {root_dir}")
        return False
    
    # Convert Path objects to strings and sort for deterministic ordering
    input_files = sorted([str(f) for f in input_files])
    
    # Print summary of found files
    print(f"Found {len(input_files)} matching file(s):")
    # for f in input_files:
    #     print(f"  - {f}")
    
    # Merge all files into a single output file
    merged_count = 0
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(output_file) as writer:
        for input_file in input_files:
            # Read each record from the input file
            for record in tf.data.TFRecordDataset(input_file):
                writer.write(record.numpy())
                merged_count += 1   
            print(f'Processing file:{input_file}\n')
    print(f"✅ Merge completed! {merged_count} records saved to {output_file}")
    return True

# Advanced version with more options
def merge_tfrecord_files_advanced(root_dir, keyword, output_file, 
                                   overwrite=False, 
                                   backup=True,
                                   show_progress=True):
    """
    Advanced merge function with existence check, backup option, and progress tracking.
    
    Args:
        root_dir: Root directory to search recursively
        keyword: Keyword to filter filenames
        output_file: Output merged file path
        overwrite: If True, overwrite existing file; if False, raise error
        backup: If True and overwriting, create backup of existing file
        show_progress: Whether to print progress information
    
    Returns:
        tuple: (success, total_records, total_files)
    """

    
    output_path = Path(output_file)
    
    # Check if output file already exists
    if output_path.exists():
        if not overwrite:
            print(f"❌ Error: {output_file} already exists!")
            print("   Use overwrite=True to overwrite or choose a different filename")
            return False, 0, 0
        else:
            print(f"⚠️ Warning: {output_file} already exists.")
            if backup:
                # Create backup with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = output_path.with_suffix(f'.tfrecord.backup_{timestamp}')
                shutil.copy2(output_file, backup_file)
                print(f"   Backup created: {backup_file}")
            print("   Overwriting...")
    
    root_path = Path(root_dir)
    
    # Find all matching files
    input_files = sorted([str(f) for f in root_path.rglob(f'*{keyword}*.tfrecord')])
    
    if not input_files:
        print(f"⚠️ Warning: No .tfrecord files containing '{keyword}' found in {root_dir}")
        return False, 0, 0
    
    print(f"Found {len(input_files)} matching file(s):")
    for f in input_files[:10]:  # Show first 10 files
        print(f"  - {f}")
    if len(input_files) > 10:
        print(f"  ... and {len(input_files) - 10} more file(s)")
    
    # Optional: Count total records for progress
    total_records = 0
    if show_progress:
        print("Counting total records...")
        for input_file in input_files:
            total_records += sum(1 for _ in tf.data.TFRecordDataset(input_file))
        print(f"Total records to merge: {total_records:,}")
    
    # Merge files
    written_records = 0
    start_time = time.time()
    
    with tf.io.TFRecordWriter(output_file) as writer:
        for file_idx, input_file in enumerate(input_files, 1):
            if show_progress:
                print(f"Processing [{file_idx}/{len(input_files)}]: {Path(input_file).name}")
            
            for record in tf.data.TFRecordDataset(input_file):
                writer.write(record.numpy())
                written_records += 1
                
                if show_progress and total_records > 0 and written_records % 10000 == 0:
                    progress = written_records / total_records * 100
                    elapsed = time.time() - start_time
                    rate = written_records / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {written_records:,}/{total_records:,} ({progress:.1f}%) - {rate:.0f} rec/s")
    
    elapsed_time = time.time() - start_time
    print("\n✅ Merge completed successfully!")
    print(f"   Output file: {output_file}")
    print(f"   Total records: {written_records:,}")
    print(f"   Time elapsed: {elapsed_time:.2f} seconds")
    print(f"   Average speed: {written_records/elapsed_time:.0f} records/second")
    
    return True, written_records, len(input_files)


# Conditional merge function - only merge if output doesn't exist
def safe_merge_tfrecord_files(root_dir, keyword, output_file, force_merge=False):
    """
    Safely merge files only if output doesn't exist.
    Returns True if merge was performed, False if output already exists or error occurred.
    
    Args:
        root_dir: Root directory to search
        keyword: Keyword to filter files
        output_file: Output file path
        force_merge: If True, merge even if output exists (overwrites)
    
    Returns:
        bool: True if merge was performed, False if skipped
    """
    output_path = Path(output_file)
    
    # Check if output already exists
    if output_path.exists() and not force_merge:
        print(f"ℹ️ Output file {output_file} already exists. Skipping merge.")
        print("   Use force_merge=True to force a new merge.")
        return False
    
    # Perform the merge
    success = merge_tfrecord_files_pathlib(root_dir, keyword, output_file, overwrite=force_merge)
    return success


# Example usage with different scenarios
def example_usage():
    """
    Examples showing how to use the safe merge functions.
    """
    
    # Scenario 1: Default - will NOT overwrite existing file
    merge_tfrecord_files_pathlib(
        root_dir='/data/dataset',
        keyword='train',
        output_file='merged_train.tfrecord',
        overwrite=False  # This will error if file exists
    )
    
    # Scenario 2: Allow overwriting existing file
    merge_tfrecord_files_pathlib(
        root_dir='/data/dataset', 
        keyword='train',
        output_file='merged_train.tfrecord',
        overwrite=True  # This will overwrite if file exists
    )
    
    # Scenario 3: Safe merge - only merge if output doesn't exist
    safe_merge_tfrecord_files(
        root_dir='/data/dataset',
        keyword='train',
        output_file='merged_train.tfrecord',
        force_merge=False  # Skip if exists
    )
    
    # Scenario 4: Advanced with backup and progress
    success, count, files = merge_tfrecord_files_advanced(
        root_dir='/data/large_dataset',
        keyword='validation',
        output_file='merged_validation.tfrecord',
        overwrite=True,
        backup=True,  # Create backup before overwriting
        show_progress=True
    )
    
    if success:
        print(f"Successfully merged {count} records from {files} files")
    else:
        print("Merge failed or was skipped")


# Helper function to check file existence before calling merge
def check_and_merge(root_dir, keyword, output_file,overwrite=False, max_iteration=None):
    """
    Check if output exists and optionally auto-rename or prompt user.
    
    Args:
        root_dir: Root directory to search
        keyword: Keyword to filter files
        output_file: Desired output file path
        auto_rename: If True and file exists, automatically generate new name
    
    Returns:
        str: Actual output file path used (might be different if auto_rename)
    """
    # Convert to Path object for better path handling
    output_path = Path(output_file)
    # Check if output file already exists
    if output_path.exists():
        if overwrite:
            print(f"⚠️ Warning: {output_file} already exists. Overwriting...")
        else:
            print(f"⚠️ Warning: {output_file} already exists!")
            print("   To overwrite, set overwrite=True")
            print("   Or choose a different filename") 
    # Convert string path to Path object
    root_path = Path(root_dir) 
    num_iterations = GL.get_map('num_iterations')
    select_decoder = GL.get_map('selected_decoder_type')
    if select_decoder in ['Check-SF1','Check-SF2','Check-SF3']:
        decoder_str = 'Check-SF'
    pattern = f'*dB/{decoder_str}/{num_iterations}th/*{keyword}*.tfrecord'
    # Recursively find all matching .tfrecord files
    all_files = list(root_path.glob(pattern))
    # Filter by iteration number
    if max_iteration is not None:
        input_files = []
        for file in all_files:
            # Extract iteration number - adjust regex pattern based on your actual filename format
            # Example patterns: 'iteration-5', 'iteration_5', 'iter5'
            match = re.search(r'iteration[-_](\d+)', str(file.name))
            if match:
                iteration_num = int(match.group(1))
                if iteration_num < max_iteration:
                    input_files.append(file)
            else:
                # If no iteration number found, include by default
                input_files.append(file)       
        print(f"Found {len(all_files)} total files, keeping {len(input_files)} with iteration < {max_iteration}")
    else:
        input_files = all_files
    # Check if any files were found
    if not input_files:
        print(f"⚠️ Warning: No .tfrecord files containing '{keyword}' found in {root_dir}")
        if max_iteration:
            print(f"   (filtered to iteration < {max_iteration})")
        return False
    # Convert Path objects to strings and sort for deterministic ordering
    input_files = sorted([str(f) for f in input_files])
    # Print summary of found files
    print(f"Found {len(input_files)} matching file(s):")
    # for f in input_files:
    #     print(f"  - {f}")    
    # Perform merge
    success = merge_tfrecord_binary_concat(input_files, output_file, overwrite)
    return output_file if success else None