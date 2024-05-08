from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory

from anthropic import Client

#config
anthropic_api_key = ""
apikey=anthropic_api_key


def count_tokens(text):
	client = Client(api_key=anthropic_api_key)
	tokens = client.count_tokens(text)
	return tokens


def load_and_prepare_documents(file_path):
    # 加载PDF文件
    loader = PyPDFDirectoryLoader(path=file_path,recursive=True,extract_images=True)
    documents = loader.load()

    return documents

def setup_llm():
    # 设置LLM
    llm = ChatAnthropic(
        anthropic_api_key= apikey,
        model='claude-3-opus-20240229', #claude-3-opus-20240229   claude-3-haiku-20240307
        temperature=0,
        max_tokens=4000
    )
    return llm

def main():
    total_tokens = 0
    max_tokens = 180000

    file_path = "./pdfs/"
    documents = load_and_prepare_documents(file_path)
    document =[]
    for doc in documents:
        doc_tokens = count_tokens(str(doc))
        # print(doc)
        if total_tokens + doc_tokens <= max_tokens:
            document.append(doc)
            total_tokens += doc_tokens
        else:
            break

    print(f"Total tokens: {total_tokens}")
    # print(chunks)
    llm = setup_llm()

    memory = ConversationBufferMemory(k=20)


    prompt_template = """{input}"""
    prompt = PromptTemplate(input_variables=["input"], template=prompt_template)



    llm_chain = LLMChain(llm=llm, prompt=prompt,memory=memory)
    # 进行问答
    while True:
        question = input("问题: ")

        prompt_v2_opus = f"""
                #####prompt engineering知识库开始#####
        {document}
        #####prompt engineering知识库结束#####
                <prompt>
                <background>
                你现在扮演一位精通prompt engineering的AI程序员专家。你的任务是运用我给你的prompt engineering的原理和方法论,优化我提供给你的prompt,使其更加清晰、简洁、无歧义。  
                </background>
                <steps>
                - 请先仔细阅读我提供的关于prompt engineering的知识库内容,这些知识将作为你优化prompt的理论基础
                - 使用XML标签尝试对我原始的prompt进行结构化,解决问题所需的角色的扮演、梳理其中的背景(background)、需要理解的要点(understanding)、沟通要求(communication)等内容
                - 分析目标任务需要经历哪些主要步骤,将其拆解成一级子任务,并用markdown列表的格式(```-```)逐条列出
                - 对于每个一级子任务,进一步分析其可能包含的二级子任务,并在对应的一级子任务下,用缩进的markdown列表(  ```-```)列出
                - 如果有必要,可以继续对二级子任务进行拆分,得到三级、四级等更细粒度的子任务,同样用缩进的markdown列表表示,但要注意避免过度拆分而失去清晰性
                - 针对每个子任务(无论级别),请思考可能遇到的异常情况或其他可能性,并提供相应的处理建议
                - 请重点关注你认为最难以完成的那个环节,运用"一步步思考"的理念,详细阐述你的思路，并在输出前结合你的思路修改prompt,直到你认为你对输出的结果满意为止  
                - 根据prompt engineering的原则对prompt进行优化,使其更加清晰、简洁、无歧义,减少完成任务过程中的阻碍
                - 分析我给你的prompt内容，找出我给你的prompt中的重点
                - 分析我给你的prompt内容，是否蕴含过程和结果2个方向，检查你给出的内容中是否暗示llm交付这些内容
                - 定义问题和目标：明确你想要LLM解决的问题或完成的任务。构建图结构：根据问题定义一个图结构，其中顶点代表问题的解决方案或中间步骤，边代表这些步骤之间的逻辑关系。生成思想：使用LLM生成与顶点相关的思想或解决方案。应用转换：对生成的思想应用转换，如聚合（Aggregate）、生成（Generate）和改进（ValidateAndImprove），以形成更复杂或更精确的解决方案。评分与排名：对当前的思想进行评分，以确定其有效性，并根据分数对思想进行排名，选择最佳输出。迭代优化：根据评分和排名的结果，可能需要回到步骤3或4，对思想进行迭代优化，直到达到满意的结果。输出最终结果：将最终选定的思想或解决方案作为输出结果
                - 请使用prompt engineering的知识所涉及的格式完整输出优化后的我给你的prompt,而不是别的prompt
                </steps>
                <exception-handling>
                在优化prompt的过程中,如果遇到对原始prompt的理解有困难、无法拆解任务步骤、找不到合适的处理异常的方法等情况,请及时向我提出,我们可以进行更多的讨论和澄清,不要勉强给出一个不完善的优化版本。 
                </exception-handling>
                <output>
                请根据上要求输出，并按照以下输出格式输出:
                ```
                1. 原始prompt的结构化版本,用XML标签表示
                2. 目标任务拆解成的多级子任务列表,用markdown格式表示
                3. 你认为最有挑战的环节是哪个,并详细说明你的思考过程
                4. 优化后的完整prompt
                ```  
                </output>
                </prompt>
                
                我的prompt内容是：
                 ```{question}```"""
      
        result = llm_chain.predict(input=prompt_v2_opus)
        print(f"答案: {result}")



if __name__ == "__main__":
    main()




