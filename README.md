# previsao-rotatividade

Hoje em dia, o termo People Analytics tem ganhado cada vez mais popularidade. Em resumo, people analytics representa um conjunto de técnicas e métodos que são orientados a dados e focados em entender os processos, comportamentos e oportunidades das pessoas dentro das organizações, guiando os times de recursos humanos nas tomadas de decisões.

Vários times de recursos humanos podem se beneficiar desse tipo de análise. Algumas abordagens mais conhecidas são o apoio no processo de seleção de talentos, estudos de performance e remuneração, e a criação de times colaborativos.

Nesse post, vou mostrar um exemplo utilizando python de como os dados podem ajudar a prever a rotatividade dentro da organização. Esse modelo consegue prever quando um funcionário pretende sair da empresa e ajuda o time de atração e seleção a se preparar para que as cadeiras não fiquem vazias! 

Vamos começar?

Aqui eu vou mostrar os principais comandos pra gerar os resultados, e mais detalhes do código podem ser vistos no arquivo Estudo-Rotatividade.ipynb.

Começamos então com a leitura da base e uma análise dos atributos que vamos utilizar pra construir o modelo :)

```
dados = pd.read_csv("turnover.csv")
dados.head()
```

A figura abaixo mostra as primeiras linhas da base (que contem informações de 15 mil funcionários diferentes). 

![Primeiras linhas da base](head.png)

Essa tabela é composta de 10 colunas diferentes:

- satisfaction: um indicador que varia de 0 a 1 e mostra o quão satisfeito o funcionário esta na emprsa.
- evaluation: um indicador da ultima performance do funcionário (varia de 0 a 1).
- number_of_projects: o número de projetos que o funcionário já participou na empresa.
- average_montly_hours: o número médio de horas que o funcionario trabalha mensalmente.
- time_spend_company: o tempo de empresa do colaborador.
- work_accident: indica 0 se o colaborador nunca teve um acidente de trabalho, e 1 caso ele teve.
- promotion: se o funcionário foi promovido nos ultimos cinco anos (1 representa promoção)
- department: o departamento do funcionário na empresa.
- salary: o nível de salário do funcionário (baixo, médio, alto).
- churn: a coluna que indica se o funcionário saiu ou não da empresa.

Nosso objetivo aqui é construir um modelo que utilize as 9 primeiras para identificar padrões de comportamento e tentar prever se o funcionário saiu ou não da empresa (a coluna churn é utilizada para ensinar o modelo, e para validar sua performance).
