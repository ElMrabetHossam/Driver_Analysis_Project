import matplotlib.pyplot as plt
import numpy as np
import os

class ReportGenerator:
    def __init__(self):
        # Utiliser 'Agg' pour ne pas avoir besoin d'affichage fenêtre (backend headless)
        plt.switch_backend('Agg')

    def generate_report(self, driver_monitor, output_filename="rapport_securite.png"):
        print("--- Génération du Rapport de Sécurité... ---")
        
        # Récupération des données
        speeds = driver_monitor.history_speed
        scores = driver_monitor.history_score
        deviations = driver_monitor.history_deviation
        ttcs = driver_monitor.history_ttc
        
        # Création des axes temporels (Frames)
        frames = range(len(speeds))
        
        # Configuration de la figure (Taille A4 Paysage approx)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Rapport d'Analyse ADAS - Trajet de {driver_monitor.total_dist:.2f} km", fontsize=16)

        # 1. Graphique VITESSE
        axs[0, 0].plot(frames, speeds, color='blue', linewidth=1)
        axs[0, 0].set_title("Profil de Vitesse (km/h)")
        axs[0, 0].set_xlabel("Frames")
        axs[0, 0].set_ylabel("km/h")
        axs[0, 0].grid(True, alpha=0.3)

        # 2. Graphique SCORE SECURITE
        axs[0, 1].plot(frames, scores, color='green', linewidth=2)
        axs[0, 1].set_title("Évolution du Score de Sécurité (%)")
        axs[0, 1].set_ylim(0, 105) # Score borné
        axs[0, 1].set_xlabel("Frames")
        axs[0, 1].grid(True, alpha=0.3)
        # Zone rouge en bas
        axs[0, 1].axhspan(0, 60, color='red', alpha=0.1)

        # 3. Graphique DEVIATION LATERALE (ZigZag)
        axs[1, 0].plot(frames, deviations, color='orange', linewidth=1)
        axs[1, 0].set_title("Stabilité Latérale (Déviation px)")
        axs[1, 0].set_xlabel("Frames")
        axs[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5) # Centre
        axs[1, 0].grid(True, alpha=0.3)

        # 4. RESUME STATISTIQUE (Texte)
        axs[1, 1].axis('off') # Pas de graphe ici, juste du texte
        
        incidents = driver_monitor.incidents
        score_final = int(driver_monitor.safety_score)
        color_grade = "green" if score_final > 80 else "orange" if score_final > 60 else "red"
        
        stats_text = (
            f"RESUME DU TRAJET\n"
            f"----------------\n\n"
            f"Distance Totale : {driver_monitor.total_dist:.2f} km\n"
            f"Vitesse Moyenne : {np.mean(speeds):.1f} km/h\n"
            f"Vitesse Max     : {np.max(speeds):.1f} km/h\n\n"
            f"INCIDENTS DETECTES\n"
            f"------------------\n"
            f"Zig-Zags / Fatigue   : {incidents['ZigZag']}\n"
            f"Freinages Brusques   : {incidents['Freinage']}\n"
            f"Alertes Collision    : {incidents['Close']}\n\n"
            f"NOTE FINALE\n"
            f"-----------\n"
            f"{score_final} / 100"
        )
        
        axs[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace', verticalalignment='center')
        
        # Sauvegarde
        output_path = os.path.join("data/processed", output_filename)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"-> Rapport généré avec succès : {output_path}")