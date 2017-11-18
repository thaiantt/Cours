******************************************************************************************
*				README INF729 Spark Trainer.scala			 *
******************************************************************************************

Pour lancer le fichier Trainer.scala : 
Ouvrir le fichier et ajouter les paths menant au fichier de données au format parquet à analyser (pour la variable df), et celui menant au fichier où l’on sauvegarde la sortie (dans model.save()).

Dans le cas où les données ne sont pas encore traitées, utiliser le script Preprocessor_cor.scala en
spécifiant le path du fichier dans les variables et en lançant ensuite le script :

> sh build_and_submit.sh Preprocessor_cor

Ouvrir un terminal, aller au path du projet, et lancer le script build_and_submit.sh de la manière suivante :

> sh build_and_submit.sh Trainer