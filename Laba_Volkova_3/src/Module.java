import java.util.Arrays;

public class Module {

    public void Module1 () {
            String moduleName;
            moduleName = "Основы Java";

            Lessons lesson = new Lessons();
            lesson.Lesson1();
            lesson.Lesson2();
            lesson.Lesson3();
    }

    public void Module2 () {
        String moduleName;
        moduleName = "Объектно-ориентированное программирование в Java";

        Lessons lesson = new Lessons();
        lesson.Lesson4();
        lesson.Lesson5();
        lesson.Lesson6();

    }

    public void Module3 () {
        String moduleName;
        moduleName = "Работа с коллекциями и файлами";

        Lessons lesson = new Lessons();
        lesson.Lesson7();
        lesson.Lesson8();

    }


}
