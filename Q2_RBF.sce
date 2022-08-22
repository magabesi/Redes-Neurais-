// Maria Gabriele Bezerra da Silva - 403736 

/* Implemente uma rede neural RBF para traçar uma superfície de separação na base de
dados two_classes.dat. Use todas as amostras para treinamento. As duas primeiras
colunas representam os atributos de cada amostra e a terceira coluna representa as
classes (+1 ou -1). */

clear
clc

base = fscanfMat('two_classes.dat');

for i = 1:2
    base(:,i) = (base(:,i)-mean(base(:,i)))/stdev(base(:,i)); // normalizando valores dos atributos
end

atributos = base (:,1:2)';  //coluna 1 e 2 atributos
classes = base(:,3)'; // coluna 3 classe

Amostras = length(atributos)/2;; // quantidade de amostras da base = 1000
At = 2; // quantidade de atributos
n_oc = 35; // quantidade de neuronios ocultos


Z=zeros(n_oc,Amostras); // Matriz Z sendo definida como matriz de 0's inicialmente
pesos = rand(At, n_oc,'normal'); // Matriz de pesos aleatórios da camada oculta, centroides

// ativação RBF  
for i=1:Amostras  // repete 1000 vezes
    for j=1:n_oc // repete 35 vezes
        Z(j,i)=exp(-norm(atributos(:,i)-pesos(:,j)).^2);      //ativando centróides
   end
end

Z =[(-1)*ones(1,1000);Z]; // Adicionar o bias = -1
W=classes*Z'*(Z*Z')^(-1); //Método dos mínimos quadrados  

// Plotagem do gráfico e curva de decisão
classe_1 = base(1:500, 1:2); //classe -1 de 1 a 500
classe_2 = base(501:1000, 1:2); //classe 1 de 501 a 1000

plot(classe_1(:, 1), classe_1(:, 2 ), 'ky'); //Classe 1 em 'y' amarelo
plot(classe_2(:, 1), classe_2(:, 2 ), 'x*'); //Classe -1 em '*' azul
legend('Classe 1','Classe -1'); //legenda 

// Determinar 1000 pontos para compor o plano    
intervalo_x1 = linspace(-1, 5, 1000);
intervalo_x2 = linspace(-1, 5, 1000);


for aux1=1:1000
    for aux2=1:1000
        for aux3 =1:n_oc
            X_new = [intervalo_x1(aux1) intervalo_x2(aux2)]' // Nova matriz de pontos  
            z_new(aux3)=exp(-norm(X_new -pesos(:,aux3)).^2); 
        end                                  
            a_new = W*[-1; z_new];          // Vetor de saídas
        
        // Superfície inconclusiva em que será traçada a superfície de decisão separando as duas classes
        if a_new < 0.001 & a_new > -0.001 then
             plot(intervalo_x1(aux1), intervalo_x2(aux2),'redd') // Plotagem da superfície de decisão, demora um pouco para aparecer 
        end
   end 
end
