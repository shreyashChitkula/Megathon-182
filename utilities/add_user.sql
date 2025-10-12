USE car_damage_detection;

INSERT INTO user_info (name, password, email, vehicle_id, contact_number, address, car_brand, model)
VALUES (
    'Prajas',
    '$2b$12$g2GjJuAFB6mBg.M53d9YQe5ygiNXO.v1V1y.9O1b1r2UZnqn4LB52',
    'prajas@gmail.com',
    'PRAJAS001',
    '9999999999',
    'Sample Address',
    'Toyota',
    'Camry'
);

SELECT 'User added successfully!' as Status;
SELECT * FROM user_info WHERE email = 'prajas@gmail.com';
