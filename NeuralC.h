#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct neuron
{
    float *pesos;
    unsigned int quant_pesos;
    float bias;
    float saida;
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

float random_float(float min, float max)
{
    float scale = rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

Neuron *create_neuron(int quant_pesos)
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

Layer *create_layer(int quant_neuronios)
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

Network *create_network(int quant_layers, int *topologia)
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
    float *saida_anterior = entrada;
    float *saida_atual = NULL;

    for (int i = 0; i < network->quant_layers; i++)
    {
        saida_atual = (float *)malloc(sizeof(float) * network->layers[i]->quant_neuronios);

        for (int j = 0; j < network->layers[i]->quant_neuronios; j++)
        {
            network->layers[i]->neuron[j]->saida = sigmoid(calcula_z(network->layers[i]->neuron[j], saida_anterior));
            saida_atual[j] = network->layers[i]->neuron[j]->saida;
        }
        if (saida_anterior != entrada)

            free(saida_anterior);

        saida_anterior = saida_atual;
    }

    free(saida_atual);
}