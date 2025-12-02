#include "NeuralC.h"
#define PARALELISMO_ATIVADO 1

void exibir_neuronio(Network *network)
{
    printf("Topologia: %d", network->topologia[0]);
    for (size_t i = 1; i < network->quant_layers + 1; i++)
    {
        printf(":%d", network->topologia[i]);
    }
    printf("\n");

    for (int i = 0; i < network->quant_layers; i++)
    {

        printf("\n____________________Camada %d______________________\n\n", i + 1);

        for (int j = 0; j < network->layers[i]->quant_neuronios; j++)
        {
            printf("____________________________________________________\n");
            printf("____________________Neuronio %d______________________\n", j + 1);
            printf("_____________________________________________________\n");

            for (int w = 0; w < network->layers[i]->neuron[j]->quant_pesos; w++)
            {
                printf("Peso %d: %f\n", w + 1, network->layers[i]->neuron[j]->pesos[w]);
            }
            printf("Bias: %f\n", network->layers[i]->neuron[j]->bias);
        }
    }
}

int main()
{
    // Treinar rede neural para aprender a porta XOR
    srand(time(NULL));
    Network *net = NULL;
    int topologia[] = {2, 10, 1};   // 2 entradas, 3 neuronios na camada oculta e 1 neuronio na camada de saída
    float target[] = {0.0, 1.0, 1.0, 0.0}; // saidas esperadas
    float taxa_aprendizado = 0.1f;
    float dados[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};

    float *entrada_treinamento[4];

    for (int i = 0; i < 4; i++)
        entrada_treinamento[i] = dados[i];

    net = create_network(2, topologia);

    definir_paralelismo(PARALELISMO_ATIVADO);
    backpropagation(net, entrada_treinamento, target, 4, taxa_aprendizado, 10000);

    forward((float[]){0.0, 0.0}, net);
    printf("Saida: %f\n", net->layers[net->quant_layers - 1]->neuron[net->layers[net->quant_layers - 1]->quant_neuronios - 1]->saida);

    destroy_network(net);
    return 0;
}