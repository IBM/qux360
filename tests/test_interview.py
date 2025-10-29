from qux360.core.interview import Interview

def test_empty_interview():
    print("hello")
    i = Interview()
    assert i.id.startswith("interview_")
    assert len(i.transcript) == 0





test_empty_interview()