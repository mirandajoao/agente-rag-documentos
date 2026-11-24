# Apostila – Introdução à Inteligência Artificial e RAG

Material de apoio da oficina "Converse com os seus documentos" (VIII SIEPEX, IFPA Campus Ananindeua). Texto escrito para iniciantes: não é preciso saber programar para acompanhar.

## Capítulo 1 – O que é Inteligência Artificial

Inteligência Artificial (IA) é a área da computação que cria sistemas capazes de realizar tarefas que, quando feitas por pessoas, exigem inteligência: reconhecer imagens, entender textos, traduzir idiomas, recomendar conteúdos ou tomar decisões a partir de dados.

Existem duas grandes formas de construir esses sistemas. Na primeira, chamada de IA simbólica, as pessoas escrevem regras explícitas, como "se a temperatura passar de 38 graus, emita um alerta". Na segunda, chamada de aprendizado de máquina (machine learning), o sistema aprende padrões a partir de exemplos, sem que alguém escreva cada regra.

Um exemplo do dia a dia: um filtro de spam aprende, a partir de milhares de e-mails marcados como "spam" ou "não spam", quais combinações de palavras costumam indicar uma mensagem indesejada. Quanto mais exemplos de boa qualidade, melhor o filtro tende a ficar.

Ponto-chave: a IA moderna aprende com dados. Por isso, a qualidade e a origem desses dados influenciam diretamente a qualidade das respostas.

## Capítulo 2 – Redes neurais e aprendizado profundo

Redes neurais artificiais são modelos matemáticos inspirados, de forma simplificada, no funcionamento dos neurônios do cérebro. Elas são formadas por camadas de "neurônios" que recebem números, fazem contas com pesos e passam o resultado adiante.

Durante o treinamento, a rede recebe um exemplo, compara a sua saída com a resposta esperada e ajusta os pesos para errar um pouco menos da próxima vez. Repetido milhões de vezes, esse processo faz a rede aprender padrões complexos.

Quando uma rede tem muitas camadas, falamos em aprendizado profundo (deep learning). É essa abordagem que está por trás do reconhecimento de voz dos celulares, da tradução automática e dos modelos de linguagem.

Ponto-chave: uma rede neural não "entende" como uma pessoa; ela aprende associações estatísticas entre entradas e saídas.

## Capítulo 3 – Modelos de linguagem (LLMs) e tokens

Um modelo de linguagem de grande porte (LLM, do inglês Large Language Model) é uma rede neural treinada com uma quantidade enorme de textos para prever qual é a próxima parte de um texto. Ao repetir essa previsão muitas vezes, o modelo consegue escrever respostas, resumos, traduções e até código.

Os LLMs não leem palavras inteiras, e sim tokens: pedaços de palavras. Em português, uma palavra costuma virar de um a três tokens. As APIs de IA cobram pelo número de tokens de entrada (o que enviamos) e de saída (o que o modelo escreve). Por isso, enviar menos texto irrelevante deixa cada pergunta mais barata.

Todo LLM tem uma data de corte: ele só conhece o que estava nos textos usados no treinamento. Ele não conhece, por exemplo, a apostila da sua turma, o contrato da sua empresa ou um regulamento publicado depois dessa data.

Ponto-chave: um LLM é ótimo com linguagem, mas o conhecimento dele é limitado ao que viu no treinamento.

## Capítulo 4 – Alucinação: quando a IA inventa

Alucinação é quando um modelo de linguagem gera uma informação falsa com aparência de verdadeira. Isso acontece porque o modelo foi treinado para produzir textos plausíveis, e não para verificar fatos.

As alucinações são mais comuns quando a pergunta trata de algo que o modelo não viu no treinamento, como números específicos, datas, nomes de pessoas pouco conhecidas ou documentos internos. Nesses casos, o modelo pode "completar a lacuna" com um valor que parece razoável.

Três atitudes ajudam a reduzir o problema: fornecer ao modelo o texto correto no momento da pergunta, pedir que ele cite de onde tirou cada informação e permitir que ele responda "não encontrei" quando não houver base para responder.

Ponto-chave: não basta a resposta parecer correta; ela precisa ter uma fonte que possa ser conferida.

## Capítulo 5 – Embeddings e busca semântica

Um embedding é uma representação de um texto como uma lista de números, chamada vetor. Modelos de embedding são treinados para que textos com significado parecido gerem vetores próximos. Assim, "carro" e "automóvel" ficam perto um do outro, mesmo sendo palavras diferentes.

Para comparar dois vetores, é comum usar a similaridade de cosseno: quanto mais próxima de 1, mais parecidos são os significados. Essa ideia permite a busca semântica, que encontra trechos pelo sentido, e não apenas pelas palavras exatas.

Os vetores ficam guardados em um banco de dados vetorial, como o ChromaDB, que encontra rapidamente os vetores mais próximos de uma pergunta. Muitas aplicações combinam essa busca com a busca por palavra-chave (como o algoritmo BM25), que é melhor para nomes, siglas e números.

Ponto-chave: embeddings transformam significado em números, e isso permite buscar por sentido.

## Capítulo 6 – RAG: geração aumentada por recuperação

RAG (Retrieval-Augmented Generation) é uma técnica que combina busca e geração de texto. Em vez de responder só com o que aprendeu, o modelo recebe, junto com a pergunta, os trechos mais relevantes de documentos escolhidos pelo usuário.

O processo tem duas fases. Na indexação, os documentos são lidos, divididos em trechos (chunks), transformados em embeddings e guardados no banco vetorial. Na consulta, a pergunta também vira um embedding, a busca traz os trechos mais parecidos e o LLM escreve a resposta usando apenas esses trechos, citando a fonte e a página.

O tamanho dos trechos importa: trechos muito pequenos perdem o contexto, e trechos muito grandes trazem texto irrelevante e encarecem cada pergunta. Uma sobreposição entre trechos vizinhos evita que uma frase importante fique cortada ao meio.

Ponto-chave: com RAG, o modelo responde com base nos seus documentos e mostra de onde tirou cada informação.

## Capítulo 7 – Agentes, segurança e uso responsável

Um agente de IA é um LLM que pode usar ferramentas, como buscar em documentos, listar arquivos ou resumir um texto inteiro, e que decide sozinho qual ferramenta usar em cada passo. O padrão ReAct organiza esse trabalho em um ciclo: pensar no que é preciso, agir com uma ferramenta, observar o resultado e só então responder.

Sistemas com IA também precisam de proteções. Uma delas é contra o prompt injection: tentativas de enganar o modelo com instruções como "ignore as suas regras", digitadas pelo usuário ou escondidas dentro de um documento. Outra é verificar se as fontes citadas na resposta realmente vieram da busca.

Por fim, há cuidados de uso responsável: não enviar documentos pessoais ou sigilosos para serviços externos sem autorização, conferir as informações importantes na fonte original e acompanhar os custos de uso das APIs.

Ponto-chave: um bom agente sabe quando responder, quando dizer que não encontrou e de onde veio cada informação.
