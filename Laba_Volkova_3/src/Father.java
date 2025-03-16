import java.util.Scanner;
import java.util.ArrayList;


public class Father extends Lessons {

    public void Manage_lesson () {
        Lessons lesson = new Lessons();
        lesson.Lesson1();

        Scanner scanner = new Scanner(System.in);

        System.out.print("Введите количество уроков: ");
        int size = scanner.nextInt();
        scanner.nextLine(); // Читаем "висящий" перевод строки

        String[] texts = new String[size];

        System.out.println("Введите " + size + " строк:");
        for (int i = 0; i < size; i++) {
            texts[i] = scanner.nextLine(); // Заполняем массив строками
        }

        System.out.println("Вы ввели:");
        for (String text : texts) {
            System.out.println(text);
        }

        scanner.close();





    }
}
