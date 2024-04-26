# Visually-Indicated-Sounds
Video-to-audio AI
- Insert Description

---
1) Audioset processing (see dedicated [repo](https://github.com/giorgiodemarchi/audioset-processing-AV)): I wrote some code to download video-audio pairs from youtube and store them on AWS S3, together with the strongly labelled annotations from AudioSet (100k+ videos).
2) [Labels augmentation with GPT](https://github.com/giorgiodemarchi/Visually-Indicated-Sounds/blob/main/GPTLabelsAugmentation.ipynb): I augment Audioset labels to identify sound emitters objects and classify as sound effect (SFX) vs ambience (AMB), by repeatetely calling OpenAI and applying majority voting.
3) I use ImageBind model ([fork repo](https://github.com/giorgiodemarchi/ImageBind)) to generate embeddings. Imagebind is a multimodal encoder that maps video, audio and text to the same embeddings space.
4) I migrate the embeddings to Pinecone generating a vector database. [Pipeline](https://github.com/giorgiodemarchi/Visually-Indicated-Sounds/blob/main/parallel_pipeline.py)
5) Semantic search
6) Eval
7) Streamlit app
