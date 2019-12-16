<?php
if($_SERVER["REQUEST_METHOD"] == "POST"){
    $file = 'data.txt';
    $current = file_get_contents($file);
    $current = $_GET['data']."\n".$current;
    file_put_contents($file, $current);
}
?>