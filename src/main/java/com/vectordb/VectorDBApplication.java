package com.vectordb;

import com.vectordb.core.DemoDataLoader;
import com.vectordb.core.OllamaClient;
import com.vectordb.core.VectorDB;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

/**
 * VectorDB — Spring Boot entry point.
 *
 * @SpringBootApplication = @Configuration + @EnableAutoConfiguration + @ComponentScan
 * Spring auto-discovers all @Component, @RestController classes under this package.
 *
 * CommandLineRunner runs after the Spring context is fully initialized —
 * the right place for startup tasks like loading demo data.
 */
@SpringBootApplication
public class VectorDBApplication {

    public static void main(String[] args) {
        SpringApplication.run(VectorDBApplication.class, args);
    }

    @Bean
    public VectorDB vectorDB() {
        return new VectorDB(16); // 16D demo vectors
    }

    @Bean
    public CommandLineRunner startup(VectorDB db, DemoDataLoader loader, OllamaClient ollama) {
        return args -> {
            // Load demo data only if the DB is empty (WAL replay may have restored data)
            if (db.size() == 0) {
                loader.load(db);
                System.out.println("Demo data loaded: " + db.size() + " vectors");
            } else {
                System.out.println("Restored " + db.size() + " vectors from WAL");
            }

            boolean ollamaUp = ollama.isAvailable();
            System.out.println("=================================");
            System.out.println("  VectorDB Engine — Java Port");
            System.out.println("  http://localhost:8080");
            System.out.println("  " + db.size() + " demo vectors | " + db.getDims() + " dims");
            System.out.println("  HNSW + KD-Tree + BruteForce");
            System.out.println("  WAL persistence: vectordb.wal");
            System.out.println("  Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
            System.out.println("=================================");
        };
    }
}
