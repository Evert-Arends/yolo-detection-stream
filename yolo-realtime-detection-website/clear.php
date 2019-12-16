<?php
$file = 'data.txt';
file_put_contents($file, '');
header('Location: index.php');