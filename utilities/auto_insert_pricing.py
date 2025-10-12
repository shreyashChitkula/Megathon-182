#!/usr/bin/env python3
"""Auto-generate and insert 17-class pricing data into database"""
import mysql.connector as connector
import config

# Base prices for 17 classes
BASE_PRICES = {
    'Bodypanel-Dent': 8000, 'Front-Windscreen-Damage': 12000,
    'Headlight-Damage': 4500, 'Rear-windscreen-Damage': 10000,
    'RunningBoard-Dent': 6000, 'Sidemirror-Damage': 3500,
    'Signlight-Damage': 2000, 'Taillight-Damage': 3000,
    'bonnet-dent': 15000, 'boot-dent': 8000,
    'doorouter-dent': 20000, 'fender-dent': 5000,
    'front-bumper-dent': 10000, 'pillar-dent': 18000,
    'quaterpanel-dent': 7000, 'rear-bumper-dent': 9000,
    'roof-dent': 22000
}

BRAND_MULTIPLIERS = {
    'MARUTI SUZUKI': 0.75, 'HYUNDAI': 1.0, 'HONDA': 1.0,
    'TOYOTA': 1.15, 'NISSAN': 1.05, 'SKODA': 1.1
}

MODEL_MULTIPLIERS = {
    'Swift': 0.8, 'Wagon R': 0.85, 'Baleno': 0.95,
    'Vitara Brezza': 1.0, 'Ertiga': 1.1, 'Grand Vitara': 1.2,
    'Amaze': 0.85, 'Jazz': 0.95, 'City': 1.0, 'WR-V': 1.05,
    'HR-V': 1.15, 'Civic': 1.15, 'CR-V': 1.25, 'Accord': 1.3, 'Pilot': 1.35,
    'Yaris': 1.0, 'Corolla': 1.15, 'Camry': 1.3, 'Innova': 1.25, 'Fortuner': 1.4,
    'i20': 1.0, 'Venue': 1.05, 'Verna': 1.1, 'Creta': 1.15, 'Tucson': 1.25,
    'Sentra': 1.0, 'Altima': 1.1, 'Pathfinder': 1.15, 'Rogue': 1.2, 'Titan': 1.25,
    'Rapid': 1.0, 'Octavia': 1.15, 'Karoq': 1.2, 'Superb': 1.3, 'Kodiaq': 1.3
}

BRANDS_MODELS = {
    'MARUTI SUZUKI': ['Swift', 'Baleno', 'Vitara Brezza', 'Wagon R', 'Ertiga', 'Grand Vitara'],
    'HONDA': ['City', 'Amaze', 'WR-V', 'Jazz', 'HR-V', 'Pilot', 'CR-V', 'Accord', 'Civic'],
    'TOYOTA': ['Corolla', 'Camry', 'Fortuner', 'Innova', 'Yaris'],
    'HYUNDAI': ['i20', 'Creta', 'Verna', 'Venue', 'Tucson'],
    'NISSAN': ['Altima', 'Rogue', 'Sentra', 'Pathfinder', 'Titan'],
    'SKODA': ['Octavia', 'Superb', 'Rapid', 'Kodiaq', 'Karoq']
}

def main():
    print("="*70)
    print("AUTO-INSERTING 17-CLASS PRICING DATA")
    print("="*70)
    
    try:
        connection = connector.connect(**config.mysql_credentials)
        cursor = connection.cursor()
        
        inserted = 0
        updated = 0
        
        for brand, models in BRANDS_MODELS.items():
            brand_mult = BRAND_MULTIPLIERS.get(brand, 1.0)
            print(f"\n Processing {brand}...")
            
            for model in models:
                model_mult = MODEL_MULTIPLIERS.get(model, 1.0)
                
                for part_name, base_price in BASE_PRICES.items():
                    price = int(base_price * brand_mult * model_mult)
                    
                    cursor.execute(
                        "SELECT COUNT(*) FROM car_models WHERE brand=%s AND model=%s AND part=%s",
                        (brand, model, part_name)
                    )
                    exists = cursor.fetchone()[0] > 0
                    
                    if exists:
                        cursor.execute(
                            "UPDATE car_models SET price=%s WHERE brand=%s AND model=%s AND part=%s",
                            (price, brand, model, part_name)
                        )
                        updated += 1
                    else:
                        cursor.execute(
                            "INSERT INTO car_models (brand, model, part, price) VALUES (%s,%s,%s,%s)",
                            (brand, model, part_name, price)
                        )
                        inserted += 1
        
        connection.commit()
        print(f"\n{'='*70}")
        print("✅ COMPLETE!")
        print(f"   Inserted: {inserted} new entries")
        print(f"   Updated: {updated} existing entries")
        print(f"   Total processed: {inserted + updated}")
        print("="*70)
        
        cursor.close()
        connection.close()
        
    except connector.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
