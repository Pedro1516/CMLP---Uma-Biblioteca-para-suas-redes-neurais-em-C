#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define SIGMOID 1
#define RELU 2
#define LEAKY_RELU 3
#define SOFTMAX 4

typedef struct neuron
{
    float *pesos;
    unsigned int quant_pesos;
    float bias;
    float saida;
    int ativacao;
} Neuron;

typedef struct layer
{
    Neuron **neuron;
    unsigned int quant_neuronios;
} Layer;

typedef struct network
{
    Layer **layers;
    unsigned int quant_layers;
    unsigned int *topologia;
} Network;

float sigmoid(float z)
{
    return 1 / (1 + exp(-z));
}

float ReLu(float z)
{
    return z > 0.0f ? z : 0.0f;
}

float Leaky_ReLu(float z, float leaky_rate)
{
    return z > 0 ? z : z * leaky_rate;
}

float random_float(float min, float max)
{
    float scale = rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

void destroy_network(Network *network)
{
    for (int i = 0; i < network->quant_layers; i++)
    {
        for (int j = 0; j < network->layers[i]->quant_neuronios; j++)
        {
            free(network->layers[i]->neuron[j]->pesos);
            free(network->layers[i]->neuron[j]);
        }
        free(network->layers[i]->neuron);
        free(network->layers[i]);
    }
    free(network->layers);
    free(network);
}

Neuron *create_neuron(unsigned int quant_pesos)
{
    Neuron *neuron = (Neuron *)malloc(sizeof(Neuron));
    neuron->bias = random_float(-1, 1);
    neuron->quant_pesos = quant_pesos;
    neuron->pesos = (float *)malloc(sizeof(float) * quant_pesos);
    neuron->saida = 0;

    for (int i = 0; i < quant_pesos; i++)
    {
        neuron->pesos[i] = random_float(-1, 1);
    }

    return neuron;
}

Layer *create_layer(unsigned int quant_neuronios)
{
    Layer *layer = (Layer *)malloc(sizeof(Layer));
    layer->quant_neuronios = quant_neuronios;
    layer->neuron = (Neuron **)malloc(sizeof(Neuron *) * quant_neuronios);
    return layer;
}

float calcula_z(Neuron *neuron, float *entradas)
{
    // z = w1x1 + w2x2 + ... + wnxn + b
    float z = neuron->bias;

    for (size_t i = 0; i < neuron->quant_pesos; i++)
    {
        z += entradas[i] * neuron->pesos[i];
    }

    return z;
}

Network *create_network(unsigned int quant_layers, unsigned int *topologia)
{
    Network *network = (Network *)malloc(sizeof(Network));
    network->layers = (Layer **)malloc(sizeof(Layer *) * quant_layers);
    network->quant_layers = quant_layers;
    network->topologia = topologia;

    for (int i = 0; i < quant_layers; i++)
    {
        network->layers[i] = create_layer(topologia[i + 1]);
        for (int j = 0; j < topologia[i + 1]; j++)
        {
            network->layers[i]->neuron[j] = create_neuron(topologia[i]);
        }
    }

    return network;
}

void forward(float *entrada, Network *network)
{
    // entrada: vetor de tamanho topologia[0]
    
    for (int i = 0; i < network->quant_layers; i++)
    {
        Layer *layer_atual = network->layers[i];

        for (unsigned int j = 0; j < layer_atual->quant_neuronios; j++)
        {
            Neuron *n = layer_atual->neuron[j];
            float z = n->bias;

            if (i == 0)
            {
                // primeira camada: entradas são os features fornecidos
                for (unsigned int p = 0; p < n->quant_pesos; p++)
                    z += n->pesos[p] * entrada[p];
            }
            else
            {
                // camadas seguintes: entradas são as saidas da camada anterior
                Layer *prev = network->layers[i - 1];
                for (unsigned int p = 0; p < n->quant_pesos; p++)
                    z += n->pesos[p] * prev->neuron[p]->saida;
            }

            n->saida = sigmoid(z);
        }
    }
}



float sigmoid_derivada(float saida_ativada)
{
    return saida_ativada * (1.0f - saida_ativada);
}

float erro_saida_1_neuronio_derivada(Network *network, float saida_esperada)
{
    return network->layers[network->quant_layers - 1]->neuron[0]->saida - saida_esperada;
}

void atualiza_pesos(Neuron *neuron, float learning_rate, float delta, float saida_anterior, unsigned int id_peso)
{
    neuron->pesos[id_peso] -= learning_rate * delta * saida_anterior;
}

void atualiza_bias(Neuron *neuron, float learning_rate, float delta)
{
    neuron->bias -= learning_rate * delta;
}

/*

void backpropagation(Network *network, float *entrada, float *target, unsigned int target_size, float learning_rate, unsigned int epoch)
{
    // Se a função de saida não for softmax(contém somente um neuronio na camada de saída)
    if (network->layers[network->quant_layers - 1]->neuron[0]->ativacao != SOFTMAX)
    {
        // calcula cada epoca
        while (epoch)
        {
            int erro_saida = 0;
            float *delta_erro_atual = (float *)calloc(network->layers[network->quant_layers - 2]->quant_neuronios, sizeof(float));
            float *delta_erro_anterior = (float *)calloc(network->layers[network->quant_layers - 1]->quant_neuronios, sizeof(float));

            for (unsigned int w = 0; w < target_size; w++)
            {
                forward(entrada, network);
                // calcula o erro da camada de saída
                erro_saida = erro_saida_1_neuronio_derivada(network, target[w]) * sigmoid_derivada(network->layers[network->quant_layers - 1]->neuron[0]->saida);

                for (unsigned int i = 0; i < network->layers[network->quant_layers - 1]->neuron[0]->quant_pesos; i++)
                {
                    atualiza_pesos(network->layers[network->quant_layers - 1]->neuron[0], learning_rate, erro_saida, network->layers[network->quant_layers - 2]->neuron[i]->saida, i);
                }
                atualiza_bias(network->layers[network->quant_layers - 1]->neuron[0], learning_rate, erro_saida);

                // propaga o erro para as camadas anteriores
                for (int i = network->quant_layers - 2; i >= 1; i--)
                {
                    for (unsigned j = 0; j < network->layers[i]->quant_neuronios; j++)
                    {
                        for (unsigned n = 0; n < network->layers[i]->neuron[j]->quant_pesos; n++)
                        {
                            if (i == network->quant_layers - 2)
                            {
                                atualiza_pesos(network->layers[i]->neuron[j], learning_rate, erro_saida, network->layers[i - 1]->neuron[n]->saida, n);
                                delta_erro_atual[j] += erro_saida * network->layers[i + 1]->neuron[j]->pesos[n] * sigmoid_derivada(network->layers[i]->neuron[j]->saida);
                            }
                            else
                            {
                                atualiza_pesos(network->layers[i]->neuron[j], learning_rate, delta_erro_anterior[j], network->layers[i - 1]->neuron[n]->saida, n);
                                delta_erro_atual[j] += delta_erro_anterior[j] * network->layers[i + 1]->neuron[j]->pesos[n] * sigmoid_derivada(network->layers[i]->neuron[j]->saida);
                            }
                        }
                        atualiza_bias(network->layers[i]->neuron[j], learning_rate, delta_erro_atual[j]);

                        memcpy(delta_erro_anterior, delta_erro_atual, sizeof(float) * network->layers[i]->quant_neuronios);
                    }
                }
            }

            epoch--;
        }
    }
}
*/
void backpropagation(Network *network, float **entrada, float *target, unsigned int target_size, float learning_rate, unsigned int epoch)
{
    const int L = network->quant_layers - 1; 

    if (network->layers[L]->quant_neuronios != 1) return;

    // Alocação (sem alterações)
    float **delta = (float **)calloc(network->quant_layers, sizeof(float *));
    for (int i = 0; i < network->quant_layers; i++) {
        delta[i] = (float *)calloc(network->layers[i]->quant_neuronios, sizeof(float));
    }

    while (epoch--) {
        
        // Loop 'w' NÃO DEVE ser paralelo (para SGD/Online training)
        for (unsigned int w = 0; w < target_size; w++) {

            forward(entrada[w], network);

            // -------- 1) DELTA SAÍDA ----------
            Neuron *out = network->layers[L]->neuron[0];
            float y = out->saida;
            float erro = y - target[w];
            delta[L][0] = erro * sigmoid_derivada(y);

            // -------- 2) DELTA OCULTAS ----------
            for (int layer = L - 1; layer >= 0; layer--) { 
                Layer *cam = network->layers[layer];
                Layer *cam_forward = network->layers[layer + 1];

                // CORREÇÃO: Paralelizar AQUI (loop j), não no loop k
                #pragma omp parallel for 
                for (unsigned int j = 0; j < cam->quant_neuronios; j++) {
                    
                    // 'soma' agora é privada para cada thread
                    float soma = 0.0f; 
                    
                    // Loop k roda sequencialmente dentro da thread (rápido e seguro)
                    for (unsigned int k = 0; k < cam_forward->quant_neuronios; k++) {
                        soma += delta[layer + 1][k] * cam_forward->neuron[k]->pesos[j];
                    }
                    
                    float s = cam->neuron[j]->saida;
                    // Escrita segura, pois cada thread tem um 'j' diferente
                    delta[layer][j] = soma * sigmoid_derivada(s);
                }
            }

            // -------- 3) ATUALIZAR PESOS ----------
            for (int layer = 0; layer <= L; layer++) { 
                Layer *cam = network->layers[layer];
                
                // CORREÇÃO: Sua lógica aqui já estava boa, mas certifique-se 
                // que 'entrada' e 'network' não estão sendo escritos.
                #pragma omp parallel for
                for (unsigned int j = 0; j < cam->quant_neuronios; j++) {
                    Neuron *n = cam->neuron[j];
                    float d = delta[layer][j];

                    for (unsigned int p = 0; p < n->quant_pesos; p++) {
                        float saida_prev;
                        if (layer == 0) saida_prev = entrada[w][p];
                        else saida_prev = network->layers[layer - 1]->neuron[p]->saida;
                        
                        // Funções que alteram APENAS o neurônio 'n' são seguras aqui
                        atualiza_pesos(n, learning_rate, d, saida_prev, p);
                    }
                    atualiza_bias(n, learning_rate, d);
                }
            }
        }
    }

    for (int i = 0; i < network->quant_layers; i++) free(delta[i]);
    free(delta);
}


void definir_paralelismo(int ativar)
{
    if (ativar)
    {
        omp_set_num_threads(omp_get_max_threads());
    }
    else
    {
        omp_set_num_threads(1);
    }
}