// Maria Gabriele Bezerra da Silva - 403736 //

/* Implemente uma rede neural Perceptron para classificar a base de dados disponível
em (https://archive.ics.uci.edu/ml/datasets/Dermatology). Nessa base as amostras com
dados faltantes devem ser removidas e os atributos devem ser normalizados por z-score.
Como estratégia de validação, utilize 50% das amostras de cada classe para treinamento
da rede neural e o restante para teste. O resultado deve ser a percentagem de acertos nas
amostras de teste. */

clear;
clc;

base = fscanfMat ('dermatology.data');      //Base com 358 elementos.

//Adicionando rotulos à base de dados.
[N c]=size(base);
r=zeros(N,6);

for i=1:N
    c=base(i,35);
    select c
        case 1 then r(i,:)=[1 0 0 0 0 0]    
        case 2 then r(i,:)=[0 1 0 0 0 0]     
        case 3 then r(i,:)=[0 0 1 0 0 0]     
        case 4 then r(i,:)=[0 0 0 1 0 0]     
        case 5 then r(i,:)=[0 0 0 0 1 0]     
        case 6 then r(i,:)=[0 0 0 0 0 1]  
    end  
end

base=[base r];  //Base com os rótulos.

for i=1:34
    base(:,i)=(base(:,i)-mean(base(:,i)))/stdev(base(:,i)); //Normalização dos atributos.
end

q=6;        //Neurônios da saída.
cont=0;     //Acertos.
epocas=2;   //Epocas para a repetição dos treinos.
p=0.01;     //Passo de aprendizado.

W=rand(q,35);   //Matriz de pesos dos neurônios.

//dividindo as classe para treino e outra para teste
x_treino = p(:,[1:55,112:141,172:206,243:266,291:314,339:348]);
D_treino = r(:,[1:55,112:141,172:206,243:266,291:314,339:348]);

x_teste = p(:,[56:110,142:171,207:241,267:290,315:338,349:358]);
D_teste = r(:,[56:110,142:171,207:241,267:290,315:338,349:358]);
