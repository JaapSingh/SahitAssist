from sahitassist.segmentation import split_paragraph_into_sentences

def test_basic_segmentation():
    para = "ਇਹ ਇੱਕ ਵਾਕ ਹੈ। ਇਹ ਦੂਜਾ ਵਾਕ ਹੈ!"
    sentences = split_paragraph_into_sentences(para)
    assert len(sentences) == 2
    assert sentences[0].endswith("।")
    assert sentences[1].endswith("!")