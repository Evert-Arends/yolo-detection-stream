<?php
// Print shit
$file = 'data.txt';
file_put_contents($file, '');
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <title>$w@g</title>
    <script src="http://code.jquery.com/jquery-latest.js"></script>
    <script>
        $(document).ready(function() {
            $("#responsecontainer").load("response.php");
            var refreshId = setInterval(function() {
                $("#responsecontainer").load('response.php');
            }, 1000);
            $.ajaxSetup({ cache: false });
        });
    </script>
</head>

<body>
<p>
    <a href="clear.php">Clear text</a>
</p>
<div id="responsecontainer">
</body>
</html>