import java.util.Scanner;

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

        NeuralNetworkManager manager = new NeuralNetworkManager(1, entradas.length * 2, 3, 2.875, entradas, saidas, epocas);

        Scanner scanner = new Scanner(System.in);
        int choice;

        do {
            System.out.println("\n--- Menu da Rede Neural ---");
            System.out.println("1. Treinar a Rede Neural");
            System.out.println("2. Carregar Modelo da Rede");
            System.out.println("3. Testar a Rede (com dados de treinamento)");
            System.out.println("4. Testar a Rede (com entrada personalizada)");
            System.out.println("5. Mostrar Parâmetros (Biases e Pesos) da Rede");
            System.out.println("0. Sair");
            System.out.print("Escolha uma opção: ");

            choice = scanner.nextInt();

            switch (choice) {
                case 1:
                    manager.treinarRede();
                    break;
                case 2:
                    System.out.print("Digite o caminho do arquivo do modelo (ex: model.txt): ");
                    String filePath = scanner.next();
                    manager.carregarRede(filePath);
                    break;
                case 3:
                    manager.testarRede();
                    break;
                case 4:
                    manager.testarComValor();
                    break;
                case 5:
                    manager.mostrarBiasEPesoModelo();
                    break;
                case 0:
                    System.out.println("Saindo...");
                    break;
                default:
                    System.out.println("Opção inválida. Por favor, tente novamente.");
            }
        } while (choice != 0);

        scanner.close();
    }
}
