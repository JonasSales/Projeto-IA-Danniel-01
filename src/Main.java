public class Main {
    public static void main(String[] args) {
        double[][] entradas = {
                {0.0}, {0.14}, {0.28}, {0.42}, {0.57}, {0.71}, {0.85}, {1.0}
        };

        double[][] saidas = {
                {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1},
                {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}
        };

        final int epocas = 2500;

        //Bons valores entradas * 2 e learning rate 2.53
        //2.875 ótimos resultados
        NeuralNetwork rede = new NeuralNetwork(1, entradas.length*2, 3, 2.875); // 1 entrada, 25 ocultos, 3 saídas
        Treinamento treinamento = new Treinamento();

        treinamento.treinar(rede, entradas, saidas, epocas);
        treinamento.resultados(rede, entradas);

    }
}
