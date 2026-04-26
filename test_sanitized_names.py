#!/usr/bin/env python
"""Test the sanitized naming logic."""

# Test pipeline name sanitization
pipeline_names = [
    "Second Test Pipeline",
    "My First Pipeline",
    "simple_pipeline",
    "Pipeline With   Multiple  Spaces"
]

print("Pipeline Name Sanitization:")
print("=" * 60)
for name in pipeline_names:
    sanitized = name.replace(" ", "_")
    print(f"'{name}' -> '{sanitized}'")

print("\n" + "=" * 60)

# Test project name sanitization
project_names = [
    "podoli",
    "spranek",
    "My Dataset",
    "Test Images 2024"
]

print("\nProject Name Sanitization:")
print("=" * 60)
for name in project_names:
    sanitized = name.replace(" ", "_")
    print(f"'{name}' -> '{sanitized}'")

print("\n" + "=" * 60)

# Test name matching (normalized)
test_cases = [
    ("podoli", "podoli", True),
    ("My Dataset", "My_Dataset", True),
    ("Test Images", "Test_Images", True),
    ("podoli", "spranek", False),
]

print("\nName Matching (with space/underscore normalization):")
print("=" * 60)
for name1, name2, expected in test_cases:
    norm1 = name1.replace(" ", "_").lower()
    norm2 = name2.replace(" ", "_").lower()
    match = norm1 == norm2
    status = "✓" if match == expected else "✗"
    print(f"{status} '{name1}' vs '{name2}': {match} (expected: {expected})")

print("\n" + "=" * 60)
print("\nExamples of resulting folder structure:")
print("=" * 60)
print("E:\\Thesis\\PipelineProjects\\")
print("  ├── Second_Test_Pipeline\\")
print("  │   ├── shared_models\\")
print("  │   ├── podoli\\          (from 'podoli' dataset)")
print("  │   └── spranek\\         (from 'spranek' dataset)")
print("  ├── My_First_Pipeline\\")
print("  │   ├── shared_models\\")
print("  │   └── My_Dataset\\      (from 'My Dataset' dataset)")
print("  └── simple_pipeline\\")
print("      ├── shared_models\\")
print("      └── Test_Images_2024\\ (from 'Test Images 2024' dataset)")
