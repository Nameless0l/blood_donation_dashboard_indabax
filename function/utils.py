def get_emoji_size(count, min_count, max_count):
            # Taille minimum et maximum des émojis
            min_size = 24
            max_size = 70
            
            # Calculer la taille proportionnelle
            if max_count == min_count:
                return (min_size + max_size) / 2
            else:
                normalized = (count - min_count) / (max_count - min_count)
                return min_size + normalized * (max_size - min_size)
        