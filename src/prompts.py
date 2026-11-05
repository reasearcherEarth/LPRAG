prompt_chunk_extraction = """
        You are required to act as an AI annotator and extract the Chunks embedded in the sentences of the provided paragraph text. 
        Below, you will be given a paragraph from a text. 
        You need to break it down sentence by sentence and extract the Chunks embedded in each sentence. 
        The extracted Chunks can be an idea, argument, or fact. 
        Each sentence may contain one or more Chunks to be extracted. 
        The extracted Chunks should be as granular as possible to ensure they cannot be further broken down.

        When extracting Chunks from a sentence, pay attention to the context within the paragraph. 
        Replace pronouns with the nouns they represent and complete any omitted sentence components to ensure the independence of the Chunks is not compromised. 
        This means that each extracted Chunk should not contain pronouns whose referents cannot be found within that Chunk.

        Below is an example interaction that can serve as a reference for format and method of extracting Chunks:

        System’s Input:
        [The Start of Paragraph text]
        Pakistan Super League (Urdu: PSL) is a Twenty20 cricket league, founded in Lahore on 9 September 2015 with five teams and now comprises six teams. Instead of operating as an association of independently owned teams, the league is a single entity in which each franchise is owned and controlled by investors.
        [The End of Paragraph text]

        Your Answer:
        [Sentence 1]
        Pakistan Super League (Urdu: PSL) is a Twenty20 cricket league, founded in Lahore on 9 September 2015 with five teams and now comprises six teams.
        [Extracted Chunks in Sentence 1]
        [Pakistan Super League (PSL) is a Twenty20 cricket league.]
        [Pakistan Super League (PSL) was founded in Lahore on 9 September 2015.]
        [Pakistan Super League (PSL) was founded with five teams.]
        [Pakistan Super League (PSL) now comprises six teams.]
        [Sentence 2]
        Instead of operating as an association of independently owned teams, the league is a single entity in which each franchise is owned and controlled by investors.
        [Extracted Chunks in Sentence 2]
        [Pakistan Super League (PSL) does not operate as an association of independently owned teams.]
        [Pakistan Super League (PSL) operates as a single entity.]
        [Each franchise in the Pakistan Super League (PSL) is owned by investors.]
        [Each franchise in the Pakistan Super League (PSL) is controlled by investors.]

        """
prompt_build_syn_query_2 = """
        You are required to act as an AI annotator and generate questions in the provided paragraph chunks. 
        Below, you will be given a bage of chunks from a paragraph. You need to generate two questions about it for each chunk. 
        The generated two questions need to be like a human made question and each chunk may contain two questions. 
        The question generated should be as similar as possible to a human question.

        When generating the two questions from a chunk, pay attention to the context within the chunk.

        Below is an example interaction that can serve as a reference for format and method of generate questions:

        System’s Input:
        [The Start bag of chunks]
        [Chunk 1]
        We propose a model-based reinforcement learning (MBRL) approach for continuous action spaces.
        [Chunk 2]
        Bthardamz is only accessible during the quest The Only Cure.
        [The End bag of chunks]

        Your Answer:
        [Questions Chunk 1]
        [What type of reinforcement learning approach was proposed?]
        [What type of action spaces does the proposed approach target?]
        [Questions Chunk 2]
        [When can Bthardamz be accessed?]
        [What quest must be completed to access Bthardamz?]
        
        """

prompt_build_syn_query = """
        You are required to act as an AI annotator and generate questions in the provided paragraph chunks. 
        Below, you will be given an chunk from a paragraph. You need to generate two questions about it. 
        The generated two questions need to be like a human made question and each chunk may contain two questions. 
        The question generated should be as similar as possible to a human question.

        When generating the two questions from a chunk, pay attention to the context within the chunk.

        Below is an example interaction that can serve as a reference for format and method of generate questions:

        System’s Input:
        [The Start of chunk]
        We propose a model-based reinforcement learning (MBRL) approach for continuous action spaces.
        [The End of chunk]

        Your Answer:
        [Question 1]
        [What type of reinforcement learning approach was proposed?]
        [Question 2]
        [What type of action spaces does the proposed approach target?]

        """

prompt_real_query_musique = """You are required to act as an AI annotator and extract the sentences that help answer a given question.
        Below, you will be given a question and a set of sentences.
        You need to read the question and extract the sentences are needed to answer the question.
        The extracted sentences have to have information to answer the question.
        The number of sentences extracted must be such that to ensure that the answer to the question is as direct as possible.
        If you cannot find a sentence that perfectly answers the question, then choose one that is more appropriate.
        DO NOT explain or justify your answer and choice, just return the sentence in the format indicated.
        Do not answer the question directly. Instead, extract the sentences that help answer the question.
        When extracting the sentences, be careful to just repeat the sentence without changing any of its content.
        Below is an example interaction that can serve as a reference for format and method of extracting sentences:

        System’s Input:
        [The Start of question]
        When was Houston Baptist University founded?
        [The End of question]

        [The Start of Sentence's set]
        The University of St. Thomas is a liberal arts college.
        The University of St. Thomas is Houston's only Catholic university.
        Rice University is a nationally recognized research university.
        Several private institutions of higher learning are located within the city of Houston.
        Rice University is located within the city of Houston.
        The University of St. Thomas is located within the city of Houston.
        Rice University has a total enrollment of slightly more than 6,000 students.
        Rice University has distinguished graduate programs.
        Rice University has research institutes.
        Rice University has the James A. Baker Institute for Public Policy.
        The James A. Baker Institute for Public Policy is a research institute at Rice University.
        Houston Baptist University is affiliated with the Baptist General Convention of Texas.
        Houston Baptist University offers bachelor's degrees.
        Houston Baptist University offers graduate degrees.
        Houston Baptist University was founded in 1960.
        In 1960 Houston Baptist University was founded.
        Hustion Baptist was founded in 60's.
        Houston Baptist University is located in the Sharpstown area.
        Houston Baptist University is located in Southwest Houston.
        [The End of Sentence's set]

        Your Format Answer:
        [Sentence 1]
        [Houston Baptist University was founded in 1960.]
        [Sentence 2]
        [In 1960 Houston Baptist University was founded]
        [Sentence 3]
        [Hustion Baptist was founded in 60's.]

        """
