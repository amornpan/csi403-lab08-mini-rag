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


class TestExercise1:
    """Test Exercise 1: Load Documents (25 points)"""
    
    def test_load_files_exists(self, student_namespace):
        assert 'load_files' in student_namespace, "Function 'load_files' not found"
    
    def test_load_files_callable(self, student_namespace):
        assert callable(student_namespace.get('load_files')), "'load_files' should be a function"


class TestExercise2:
    """Test Exercise 2: Chunk Text (25 points)"""
    
    def test_make_chunks_exists(self, student_namespace):
        assert 'make_chunks' in student_namespace, "Function 'make_chunks' not found"
    
    def test_make_chunks_callable(self, student_namespace):
        assert callable(student_namespace.get('make_chunks')), "'make_chunks' should be a function"
    
    def test_make_chunks_works(self, student_namespace):
        func = student_namespace.get('make_chunks')
        if func:
            result = func('ABCDEFGHIJ', 5)
            assert result == ['ABCDE', 'FGHIJ'], f"make_chunks('ABCDEFGHIJ', 5) should return ['ABCDE', 'FGHIJ'], got {result}"


class TestExercise3:
    """Test Exercise 3: Search Function (25 points)"""
    
    def test_find_matches_exists(self, student_namespace):
        assert 'find_matches' in student_namespace, "Function 'find_matches' not found"
    
    def test_find_matches_callable(self, student_namespace):
        assert callable(student_namespace.get('find_matches')), "'find_matches' should be a function"
    
    def test_find_matches_works(self, student_namespace):
        func = student_namespace.get('find_matches')
        if func:
            test_chunks = ['fever and rash', 'diarrhea symptoms', 'high fever']
            result = func('fever', test_chunks)
            assert len(result) == 2, f"find_matches('fever', test_chunks) should return 2 matches, got {len(result)}"


class TestExercise4:
    """Test Exercise 4: Complete System (25 points)"""
    
    def test_minirag_exists(self, student_namespace):
        assert 'MiniRAG' in student_namespace, "Class 'MiniRAG' not found"
    
    def test_minirag_is_class(self, student_namespace):
        assert isinstance(student_namespace.get('MiniRAG'), type), "'MiniRAG' should be a class"
    
    def test_minirag_has_methods(self, student_namespace):
        MiniRAG = student_namespace.get('MiniRAG')
        if MiniRAG:
            rag = MiniRAG()
            assert hasattr(rag, 'load'), "MiniRAG should have 'load' method"
            assert hasattr(rag, 'index'), "MiniRAG should have 'index' method"
            assert hasattr(rag, 'search'), "MiniRAG should have 'search' method"
    
    def test_minirag_search_works(self, student_namespace):
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
