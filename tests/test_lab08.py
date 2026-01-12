"""
Lab 08: Mini RAG System - Auto-grading Tests
"""

import pytest
import os
import nbformat

# Get paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(BASE_DIR, 'exercise', 'Lab08_Exercise.ipynb')


@pytest.fixture(scope="session")
def student_namespace():
    """Execute student notebook and return namespace with variables."""
    
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    namespace = {'__name__': '__main__'}
    
    original_dir = os.getcwd()
    exercise_dir = os.path.join(BASE_DIR, 'exercise')
    os.chdir(exercise_dir)
    
    try:
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if '# Quick check' in cell.source or '# Verification' in cell.source:
                    continue
                try:
                    exec(cell.source, namespace)
                except Exception as e:
                    print(f"Cell execution warning: {e}")
    finally:
        os.chdir(original_dir)
    
    return namespace


# ============== Exercise 1: Load Documents (25 points) ==============

def test_ex1_load_files_exists(student_namespace):
    """Test that load_files function exists"""
    assert 'load_files' in student_namespace, "Function 'load_files' not found"


def test_ex1_load_files_callable(student_namespace):
    """Test that load_files is callable"""
    assert callable(student_namespace.get('load_files')), "'load_files' should be a function"


# ============== Exercise 2: Chunk Text (25 points) ==============

def test_ex2_make_chunks_exists(student_namespace):
    """Test that make_chunks function exists"""
    assert 'make_chunks' in student_namespace, "Function 'make_chunks' not found"


def test_ex2_make_chunks_callable(student_namespace):
    """Test that make_chunks is callable"""
    assert callable(student_namespace.get('make_chunks')), "'make_chunks' should be a function"


def test_ex2_make_chunks_works(student_namespace):
    """Test that make_chunks works correctly"""
    func = student_namespace.get('make_chunks')
    if func:
        result = func('ABCDEFGHIJ', 5)
        assert result == ['ABCDE', 'FGHIJ'], f"make_chunks('ABCDEFGHIJ', 5) should return ['ABCDE', 'FGHIJ'], got {result}"


# ============== Exercise 3: Search Function (25 points) ==============

def test_ex3_find_matches_exists(student_namespace):
    """Test that find_matches function exists"""
    assert 'find_matches' in student_namespace, "Function 'find_matches' not found"


def test_ex3_find_matches_callable(student_namespace):
    """Test that find_matches is callable"""
    assert callable(student_namespace.get('find_matches')), "'find_matches' should be a function"


def test_ex3_find_matches_works(student_namespace):
    """Test that find_matches works correctly"""
    func = student_namespace.get('find_matches')
    if func:
        test_chunks = ['fever and rash', 'diarrhea symptoms', 'high fever']
        result = func('fever', test_chunks)
        assert len(result) == 2, f"find_matches('fever', test_chunks) should return 2 matches, got {len(result)}"


# ============== Exercise 4: Complete System (25 points) ==============

def test_ex4_minirag_exists(student_namespace):
    """Test that MiniRAG class exists"""
    assert 'MiniRAG' in student_namespace, "Class 'MiniRAG' not found"


def test_ex4_minirag_is_class(student_namespace):
    """Test that MiniRAG is a class"""
    assert isinstance(student_namespace.get('MiniRAG'), type), "'MiniRAG' should be a class"


def test_ex4_minirag_has_methods(student_namespace):
    """Test that MiniRAG has required methods"""
    MiniRAG = student_namespace.get('MiniRAG')
    if MiniRAG:
        rag = MiniRAG()
        assert hasattr(rag, 'load'), "MiniRAG should have 'load' method"
        assert hasattr(rag, 'index'), "MiniRAG should have 'index' method"
        assert hasattr(rag, 'search'), "MiniRAG should have 'search' method"


def test_ex4_minirag_search_works(student_namespace):
    """Test that MiniRAG search works"""
    MiniRAG = student_namespace.get('MiniRAG')
    if MiniRAG:
        rag = MiniRAG()
        rag.chunks = [
            'Rubella causes fever and rash',
            'Cholera causes diarrhea',
            'Dengue fever is dangerous'
        ]
        results = rag.search('fever')
        assert results is not None, "search() should return results"
        if results:
            assert len(results) == 2, f"search('fever') should find 2 chunks, got {len(results)}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
